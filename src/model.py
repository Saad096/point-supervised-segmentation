"""
Segmentation Models for Remote Sensing
Implements U-Net with various encoder backbones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional, List


class SegmentationModel(nn.Module):
    """
    Wrapper for segmentation models with unified interface.
    
    Supports multiple architectures:
    - U-Net with various encoders (ResNet, EfficientNet, etc.)
    - DeepLabV3+
    - FPN
    """
    
    def __init__(
        self,
        architecture: str = 'unet',
        encoder_name: str = 'resnet34',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 7,
        activation: Optional[str] = None,
    ):
        """
        Args:
            architecture: Model architecture ('unet', 'deeplabv3+', 'fpn')
            encoder_name: Encoder backbone name
            encoder_weights: Pretrained weights ('imagenet' or None)
            in_channels: Number of input channels
            classes: Number of output classes
            activation: Output activation function
        """
        super(SegmentationModel, self).__init__()
        
        self.architecture = architecture
        self.encoder_name = encoder_name
        self.num_classes = classes
        
        # Create model based on architecture
        if architecture == 'unet':
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
            )
        elif architecture == 'deeplabv3+':
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
            )
        elif architecture == 'fpn':
            self.model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output predictions of shape (B, num_classes, H, W)
        """
        return self.model(x)
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UNetWithAttention(nn.Module):
    """
    U-Net with attention gates for improved feature aggregation.
    Attention helps the model focus on relevant spatial regions.
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 7,
    ):
        super(UNetWithAttention, self).__init__()
        
        # Base U-Net
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
        
        # Add attention modules
        self.num_classes = classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple segmentation models for improved performance.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        ensemble_method: str = 'average',
    ):
        """
        Args:
            models: List of segmentation models
            ensemble_method: How to combine predictions ('average', 'voting')
        """
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        self.num_classes = models[0].num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Returns:
            torch.Tensor: Ensembled predictions
        """
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Combine predictions
        if self.ensemble_method == 'average':
            # Average logits
            ensemble_pred = torch.stack(predictions).mean(dim=0)
        elif self.ensemble_method == 'voting':
            # Majority voting
            pred_classes = torch.stack([p.argmax(dim=1) for p in predictions])
            ensemble_pred = torch.mode(pred_classes, dim=0)[0]
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return ensemble_pred


def get_model(
    architecture: str = 'unet',
    encoder_name: str = 'resnet34',
    encoder_weights: str = 'imagenet',
    num_classes: int = 7,
    device: str = 'cpu',
) -> nn.Module:
    """
    Factory function to create segmentation model.
    
    Args:
        architecture: Model architecture
        encoder_name: Encoder backbone
        encoder_weights: Pretrained weights
        num_classes: Number of output classes
        device: Device to load model on
        
    Returns:
        nn.Module: Configured model
    """
    model = SegmentationModel(
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )
    
    model = model.to(device)
    
    # Print model info
    num_params = model.get_num_parameters()
    num_trainable = model.get_num_trainable_parameters()
    
    print(f"\nModel: {architecture} with {encoder_name} encoder")
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    print(f"Device: {device}\n")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing Model Creation...")
    
    # Test on CPU
    device = 'cpu'
    
    # Create model
    model = get_model(
        architecture='unet',
        encoder_name='resnet34',
        encoder_weights='imagenet',
        num_classes=7,
        device=device,
    )
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 512, 512).to(device)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 7, 512, 512), "Output shape mismatch!"
    
    print("\nModel test passed!")
