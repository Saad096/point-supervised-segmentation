"""
Evaluation Metrics for Semantic Segmentation
Implements IoU, F1, Precision, Recall, and Accuracy
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class SegmentationMetrics:
    """
    Comprehensive metrics for semantic segmentation evaluation.
    
    Computes:
    - Pixel Accuracy
    - Mean Accuracy (per-class accuracy)
    - IoU (Intersection over Union) per class and mean
    - F1 Score per class and mean
    - Precision and Recall per class
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        ignore_index: int = -1,
    ):
        """
        Args:
            num_classes: Number of classes
            class_names: List of class names for visualization
            ignore_index: Index to ignore in metrics computation
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        self.ignore_index = ignore_index
        
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.confusion_mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_samples = 0
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Update metrics with new batch of predictions.
        
        Args:
            predictions: Model predictions of shape (B, C, H, W) or (B, H, W)
            targets: Ground truth labels of shape (B, H, W)
        """
        # Convert predictions to class indices if needed
        if predictions.dim() == 4:
            predictions = predictions.argmax(dim=1)
        
        # Move to CPU and convert to numpy
        predictions = predictions.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        
        # Create mask for valid pixels (not ignore_index)
        valid_mask = targets != self.ignore_index
        
        if not valid_mask.any():
            return
        
        # Filter predictions and targets
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        # Update confusion matrix
        cm = confusion_matrix(
            targets,
            predictions,
            labels=list(range(self.num_classes))
        )
        
        self.confusion_mat += cm
        self.total_samples += len(targets)
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics based on accumulated confusion matrix.
        
        Returns:
            dict: Dictionary containing all computed metrics
        """
        cm = self.confusion_mat
        
        # True Positives, False Positives, False Negatives
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)
        
        # Pixel Accuracy
        pixel_acc = tp.sum() / (cm.sum() + 1e-10)
        
        # Per-class metrics
        iou_per_class = tp / (tp + fp + fn + 1e-10)
        precision_per_class = tp / (tp + fp + 1e-10)
        recall_per_class = tp / (tp + fn + 1e-10)
        f1_per_class = 2 * (precision_per_class * recall_per_class) / \
                      (precision_per_class + recall_per_class + 1e-10)
        accuracy_per_class = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        
        # Mean metrics
        mean_iou = np.nanmean(iou_per_class)
        mean_f1 = np.nanmean(f1_per_class)
        mean_precision = np.nanmean(precision_per_class)
        mean_recall = np.nanmean(recall_per_class)
        mean_acc = np.nanmean(accuracy_per_class)
        
        # Compile metrics
        metrics = {
            'pixel_accuracy': pixel_acc,
            'mean_accuracy': mean_acc,
            'mean_iou': mean_iou,
            'mean_f1': mean_f1,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
        }
        
        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f'iou_{class_name}'] = iou_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get the accumulated confusion matrix."""
        return self.confusion_mat.copy()
    
    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        normalize: bool = False,
    ) -> plt.Figure:
        """
        Plot confusion matrix as heatmap.
        
        Args:
            save_path: Path to save the figure
            normalize: Whether to normalize the confusion matrix
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        cm = self.confusion_mat.copy()
        
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_class_metrics(
        self,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot per-class metrics as bar charts.
        
        Args:
            save_path: Path to save the figure
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        metrics = self.compute_metrics()
        
        # Extract per-class metrics
        iou_scores = [metrics[f'iou_{name}'] for name in self.class_names]
        f1_scores = [metrics[f'f1_{name}'] for name in self.class_names]
        precision_scores = [metrics[f'precision_{name}'] for name in self.class_names]
        recall_scores = [metrics[f'recall_{name}'] for name in self.class_names]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # IoU
        axes[0, 0].bar(self.class_names, iou_scores, color='steelblue')
        axes[0, 0].set_title('IoU per Class', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('IoU Score', fontsize=10)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].axhline(y=metrics['mean_iou'], color='r', linestyle='--', 
                           label=f"Mean: {metrics['mean_iou']:.3f}")
        axes[0, 0].legend()
        
        # F1 Score
        axes[0, 1].bar(self.class_names, f1_scores, color='darkorange')
        axes[0, 1].set_title('F1 Score per Class', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('F1 Score', fontsize=10)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].axhline(y=metrics['mean_f1'], color='r', linestyle='--',
                          label=f"Mean: {metrics['mean_f1']:.3f}")
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].bar(self.class_names, precision_scores, color='forestgreen')
        axes[1, 0].set_title('Precision per Class', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Precision', fontsize=10)
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].axhline(y=metrics['mean_precision'], color='r', linestyle='--',
                          label=f"Mean: {metrics['mean_precision']:.3f}")
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].bar(self.class_names, recall_scores, color='crimson')
        axes[1, 1].set_title('Recall per Class', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Recall', fontsize=10)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].axhline(y=metrics['mean_recall'], color='r', linestyle='--',
                          label=f"Mean: {metrics['mean_recall']:.3f}")
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class metrics plot saved to {save_path}")
        
        return fig


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> torch.Tensor:
    """
    Compute IoU (Intersection over Union) for each class.
    
    Args:
        pred: Predictions of shape (B, H, W)
        target: Ground truth of shape (B, H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore
        
    Returns:
        torch.Tensor: IoU scores of shape (num_classes,)
    """
    ious = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        # Create mask for valid pixels
        valid_mask = (target != ignore_index)
        
        pred_cls = pred_cls & valid_mask
        target_cls = target_cls & valid_mask
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        iou = intersection / (union + 1e-10)
        ious.append(iou)
    
    return torch.stack(ious)


if __name__ == "__main__":
    # Test metrics
    print("Testing Segmentation Metrics...")
    
    # Create dummy data
    batch_size = 4
    num_classes = 7
    height, width = 256, 256
    
    predictions = torch.randint(0, num_classes, (batch_size, height, width))
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Create metrics tracker
    class_names = [f"Class_{i}" for i in range(num_classes)]
    metrics = SegmentationMetrics(num_classes, class_names)
    
    # Update metrics
    metrics.update(predictions, targets)
    
    # Compute metrics
    results = metrics.compute_metrics()
    
    print("\nComputed Metrics:")
    print(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}")
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    print(f"Mean F1: {results['mean_f1']:.4f}")
    
    print("\nMetrics test passed!")
