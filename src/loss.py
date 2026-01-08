"""
Partial Cross-Entropy Loss for Point-Based Supervision
Implements loss function that only computes CE on labeled points
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PartialCrossEntropyLoss(nn.Module):
    """
    Partial Cross-Entropy Loss for weakly supervised segmentation with point annotations.
    
    This loss only computes the cross-entropy on pixels where labels are available (points),
    ignoring unlabeled regions. This enables training segmentation networks with sparse
    point annotations instead of dense pixel-wise masks.
    
    Args:
        ignore_index (int): Index to ignore in the target (unlabeled pixels)
        weight (Optional[torch.Tensor]): Class weights for handling imbalanced data
        reduction (str): Specifies the reduction to apply: 'none' | 'mean' | 'sum'
        lambda_point (float): Weight for point supervision loss
        label_smoothing (float): Label smoothing factor (0.0 = no smoothing)
    """
    
    def __init__(
        self,
        ignore_index: int = -1,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        lambda_point: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        super(PartialCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduction = reduction
        self.lambda_point = lambda_point
        self.label_smoothing = label_smoothing
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute partial cross-entropy loss.
        
        Args:
            predictions (torch.Tensor): Model predictions of shape (B, C, H, W)
                where B=batch, C=classes, H=height, W=width
            targets (torch.Tensor): Sparse point labels of shape (B, H, W)
                where labeled pixels have class indices [0, C-1]
                and unlabeled pixels have ignore_index value
                
        Returns:
            torch.Tensor: Computed loss value
        """
        # Get dimensions
        batch_size, num_classes, height, width = predictions.shape
        
        # Create mask for labeled points (where target != ignore_index)
        labeled_mask = (targets != self.ignore_index)
        
        # Check if there are any labeled points
        if not labeled_mask.any():
            # Return zero loss if no labeled points
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Flatten predictions and targets
        predictions_flat = predictions.permute(0, 2, 3, 1).reshape(-1, num_classes)
        targets_flat = targets.reshape(-1)
        labeled_mask_flat = labeled_mask.reshape(-1)
        
        # Extract only labeled predictions and targets
        labeled_predictions = predictions_flat[labeled_mask_flat]
        labeled_targets = targets_flat[labeled_mask_flat]
        
        # Compute cross-entropy loss with optional label smoothing
        if self.label_smoothing > 0:
            loss = self._cross_entropy_with_label_smoothing(
                labeled_predictions,
                labeled_targets,
                num_classes
            )
        else:
            loss = F.cross_entropy(
                labeled_predictions,
                labeled_targets,
                weight=self.weight,
                reduction=self.reduction
            )
        
        # Apply point supervision weight
        loss = self.lambda_point * loss
        
        return loss
    
    def _cross_entropy_with_label_smoothing(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss with label smoothing.
        
        Label smoothing helps prevent overconfident predictions by
        distributing some probability mass to other classes.
        """
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes).float()
        
        # Apply label smoothing
        targets_smooth = targets_one_hot * (1 - self.label_smoothing) + \
                        self.label_smoothing / num_classes
        
        # Compute log softmax
        log_probs = F.log_softmax(predictions, dim=1)
        
        # Compute loss
        loss = -(targets_smooth * log_probs).sum(dim=1)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
            
        return loss


class PartialCrossEntropyWithConsistency(nn.Module):
    """
    Extended Partial CE Loss with consistency regularization.
    
    This loss combines:
    1. Partial CE on labeled points
    2. Consistency regularization on unlabeled regions
    
    The consistency term encourages smooth predictions in unlabeled regions,
    helping the model generalize better from sparse point annotations.
    
    Args:
        ignore_index (int): Index to ignore in the target
        weight (Optional[torch.Tensor]): Class weights
        lambda_point (float): Weight for point supervision
        lambda_consistency (float): Weight for consistency regularization
        consistency_type (str): Type of consistency: 'entropy' | 'variance'
    """
    
    def __init__(
        self,
        ignore_index: int = -1,
        weight: Optional[torch.Tensor] = None,
        lambda_point: float = 1.0,
        lambda_consistency: float = 0.1,
        consistency_type: str = 'entropy',
    ):
        super(PartialCrossEntropyWithConsistency, self).__init__()
        self.partial_ce = PartialCrossEntropyLoss(
            ignore_index=ignore_index,
            weight=weight,
            lambda_point=lambda_point
        )
        self.lambda_consistency = lambda_consistency
        self.consistency_type = consistency_type
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined loss with consistency regularization.
        
        Returns:
            tuple: (total_loss, loss_dict) where loss_dict contains individual components
        """
        # Compute partial CE loss
        ce_loss = self.partial_ce(predictions, targets)
        
        # Compute consistency loss on unlabeled regions
        unlabeled_mask = (targets == self.partial_ce.ignore_index)
        
        if unlabeled_mask.any():
            if self.consistency_type == 'entropy':
                consistency_loss = self._entropy_consistency(predictions, unlabeled_mask)
            else:
                consistency_loss = self._variance_consistency(predictions, unlabeled_mask)
            
            consistency_loss = self.lambda_consistency * consistency_loss
        else:
            consistency_loss = torch.tensor(0.0, device=predictions.device)
        
        # Total loss
        total_loss = ce_loss + consistency_loss
        
        # Return loss components for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'consistency_loss': consistency_loss.item(),
        }
        
        return total_loss, loss_dict
    
    def _entropy_consistency(
        self,
        predictions: torch.Tensor,
        unlabeled_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Entropy-based consistency: encourage confident predictions in unlabeled regions.
        Lower entropy = more confident predictions.
        """
        # Compute softmax probabilities
        probs = F.softmax(predictions, dim=1)
        
        # Compute entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        
        # Apply mask and compute mean
        masked_entropy = entropy * unlabeled_mask.float()
        loss = masked_entropy.sum() / (unlabeled_mask.sum() + 1e-8)
        
        return loss
    
    def _variance_consistency(
        self,
        predictions: torch.Tensor,
        unlabeled_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Variance-based consistency: encourage low variance in spatial neighborhoods.
        """
        # Compute softmax probabilities
        probs = F.softmax(predictions, dim=1)
        
        # Compute spatial variance using average pooling
        avg_probs = F.avg_pool2d(probs, kernel_size=3, stride=1, padding=1)
        variance = ((probs - avg_probs) ** 2).mean(dim=1)
        
        # Apply mask and compute mean
        masked_variance = variance * unlabeled_mask.float()
        loss = masked_variance.sum() / (unlabeled_mask.sum() + 1e-8)
        
        return loss


def get_loss_function(config: dict) -> nn.Module:
    """
    Factory function to create loss function based on configuration.
    
    Args:
        config (dict): Loss configuration dictionary
        
    Returns:
        nn.Module: Configured loss function
    """
    loss_type = config.get('type', 'partial_ce')
    
    if loss_type == 'partial_ce':
        return PartialCrossEntropyLoss(
            ignore_index=config.get('ignore_index', -1),
            weight=config.get('weight', None),
            lambda_point=config.get('lambda_point', 1.0),
        )
    elif loss_type == 'partial_ce_consistency':
        return PartialCrossEntropyWithConsistency(
            ignore_index=config.get('ignore_index', -1),
            weight=config.get('weight', None),
            lambda_point=config.get('lambda_point', 1.0),
            lambda_consistency=config.get('lambda_consistency', 0.1),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test the loss function
    print("Testing Partial Cross-Entropy Loss...")
    
    # Create dummy data
    batch_size, num_classes, height, width = 2, 7, 256, 256
    predictions = torch.randn(batch_size, num_classes, height, width)
    
    # Create sparse point labels (most pixels are unlabeled)
    targets = torch.full((batch_size, height, width), -1, dtype=torch.long)
    
    # Simulate 100 labeled points per image
    for b in range(batch_size):
        for _ in range(100):
            h, w = torch.randint(0, height, (1,)), torch.randint(0, width, (1,))
            targets[b, h, w] = torch.randint(0, num_classes, (1,))
    
    # Test basic loss
    loss_fn = PartialCrossEntropyLoss()
    loss = loss_fn(predictions, targets)
    print(f"Basic Partial CE Loss: {loss.item():.4f}")
    
    # Test loss with consistency
    loss_fn_cons = PartialCrossEntropyWithConsistency()
    total_loss, loss_dict = loss_fn_cons(predictions, targets)
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    
    print("\nAll tests passed!")
