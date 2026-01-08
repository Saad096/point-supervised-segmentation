"""
Evaluation Script for Point-Supervised Segmentation
Tests trained models and generates visualizations
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from typing import List, Tuple, Optional

sys.path.append(str(Path(__file__).parent))

from model import get_model
from data_loader import DeepGlobeDataset, get_transforms
from metrics import SegmentationMetrics


class ModelEvaluator:
    """
    Evaluator for trained segmentation models.
    
    Provides:
    - Model evaluation on test set
    - Prediction visualization
    - Performance analysis
    """
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str,
        class_names: List[str],
        class_colors: List[Tuple[int, int, int]],
        output_dir: Path,
    ):
        """
        Args:
            model: Trained segmentation model
            test_loader: Test data loader
            device: Device to run on
            class_names: List of class names
            class_colors: List of RGB colors for visualization
            output_dir: Directory to save results
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        self.class_colors = class_colors
        self.output_dir = output_dir
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metrics tracker
        self.metrics = SegmentationMetrics(
            num_classes=len(class_names),
            class_names=class_names,
            ignore_index=-1,
        )
    
    @torch.no_grad()
    def evaluate(self) -> dict:
        """
        Evaluate model on test set.
        
        Returns:
            dict: Evaluation metrics
        """
        print("\nEvaluating model on test set...")
        
        self.model.eval()
        self.metrics.reset()
        
        all_predictions = []
        all_targets = []
        
        for images, masks in tqdm(self.test_loader, desc="Testing"):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            predictions = self.model(images)
            
            # Update metrics
            self.metrics.update(predictions, masks)
            
            # Store for visualization
            pred_classes = predictions.argmax(dim=1).cpu().numpy()
            all_predictions.append(pred_classes)
            all_targets.append(masks.cpu().numpy())
        
        # Compute metrics
        metrics = self.metrics.compute_metrics()
        
        # Print results
        print("\n" + "="*80)
        print("Test Set Evaluation Results")
        print("="*80)
        print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        print(f"Mean Accuracy: {metrics['mean_accuracy']:.4f}")
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"Mean F1: {metrics['mean_f1']:.4f}")
        print(f"Mean Precision: {metrics['mean_precision']:.4f}")
        print(f"Mean Recall: {metrics['mean_recall']:.4f}")
        
        print("\nPer-Class IoU:")
        for class_name in self.class_names:
            iou = metrics[f'iou_{class_name}']
            print(f"  {class_name:20s}: {iou:.4f}")
        print("="*80 + "\n")
        
        # Save metrics
        metrics_file = self.output_dir / 'test_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to {metrics_file}")
        
        return metrics
    
    def visualize_predictions(
        self,
        num_samples: int = 10,
        save_individual: bool = True,
    ):
        """
        Visualize model predictions.
        
        Args:
            num_samples: Number of samples to visualize
            save_individual: Whether to save individual predictions
        """
        print(f"\nGenerating visualizations for {num_samples} samples...")
        
        self.model.eval()
        
        vis_dir = self.output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        sample_count = 0
        
        for batch_idx, (images, masks) in enumerate(self.test_loader):
            if sample_count >= num_samples:
                break
            
            images = images.to(self.device)
            
            # Get predictions
            with torch.no_grad():
                predictions = self.model(images)
                pred_classes = predictions.argmax(dim=1)
            
            # Move to CPU for visualization
            images = images.cpu()
            masks = masks.cpu()
            pred_classes = pred_classes.cpu()
            
            # Visualize each image in batch
            for i in range(images.shape[0]):
                if sample_count >= num_samples:
                    break
                
                # Create visualization
                fig = self.create_visualization(
                    images[i],
                    masks[i],
                    pred_classes[i],
                    sample_idx=sample_count
                )
                
                # Save
                if save_individual:
                    save_path = vis_dir / f'sample_{sample_count:03d}.png'
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                
                sample_count += 1
        
        print(f"Visualizations saved to {vis_dir}")
    
    def create_visualization(
        self,
        image: torch.Tensor,
        target: torch.Tensor,
        prediction: torch.Tensor,
        sample_idx: int,
    ) -> plt.Figure:
        """
        Create a visualization comparing input, ground truth, and prediction.
        
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        image_np = image.permute(1, 2, 0).numpy()
        image_np = std * image_np + mean
        image_np = np.clip(image_np, 0, 1)
        
        # Convert masks to colored images
        target_colored = self.mask_to_color(target.numpy())
        pred_colored = self.mask_to_color(prediction.numpy())
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(image_np)
        axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(target_colored)
        axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(pred_colored)
        axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        # Add legend
        self.add_legend(fig)
        
        plt.suptitle(f'Sample {sample_idx}', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    def mask_to_color(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert class indices to RGB image.
        
        Args:
            mask: Mask of shape (H, W) with class indices
            
        Returns:
            np.ndarray: RGB image of shape (H, W, 3)
        """
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_idx, color in enumerate(self.class_colors):
            colored[mask == class_idx] = color
        
        return colored
    
    def add_legend(self, fig: plt.Figure):
        """Add legend with class colors to figure."""
        from matplotlib.patches import Patch
        
        legend_elements = [
            Patch(
                facecolor=np.array(color) / 255.0,
                label=name
            )
            for name, color in zip(self.class_names, self.class_colors)
        ]
        
        fig.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=len(self.class_names),
            bbox_to_anchor=(0.5, -0.05),
            fontsize=10,
            frameon=True,
        )
    
    def generate_report(self):
        """Generate comprehensive evaluation report."""
        print("\nGenerating evaluation report...")
        
        # Plot confusion matrix
        cm_path = self.output_dir / 'confusion_matrix.png'
        self.metrics.plot_confusion_matrix(save_path=cm_path, normalize=True)
        
        # Plot class metrics
        metrics_path = self.output_dir / 'class_metrics.png'
        self.metrics.plot_class_metrics(save_path=metrics_path)
        
        print(f"Report generated in {self.output_dir}")


def evaluate_checkpoint(
    checkpoint_path: str,
    data_dir: str,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
    num_visualizations: int = 10,
):
    """
    Evaluate a trained model checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Path to dataset
        output_dir: Directory to save results (auto-generate if None)
        device: Device to run on (auto-detect if None)
        num_visualizations: Number of samples to visualize
    """
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nLoading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create output directory
    if output_dir is None:
        checkpoint_dir = Path(checkpoint_path).parent
        output_dir = checkpoint_dir / 'evaluation'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = get_model(
        architecture=config['architecture'],
        encoder_name=config['encoder'],
        encoder_weights=None,  # Don't load pretrained weights
        num_classes=config['num_classes'],
        device=device,
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully (epoch {checkpoint['epoch']})")
    
    # Create test loader
    transform = get_transforms('test', config['image_size'])
    
    test_dataset = DeepGlobeDataset(
        data_dir=data_dir,
        split='test',
        image_size=config['image_size'],
        point_sampler=None,  # Use dense masks for testing
        transform=transform,
        use_point_labels=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=config['class_names'],
        class_colors=config.get('class_colors', [(255, 0, 0)] * config['num_classes']),
        output_dir=output_dir,
    )
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Generate visualizations
    evaluator.visualize_predictions(num_samples=num_visualizations)
    
    # Generate report
    evaluator.generate_report()
    
    print(f"\n{'='*80}")
    print("Evaluation Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/raw',
        help='Path to dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default=None,
        help='Device to run on'
    )
    parser.add_argument(
        '--num_vis',
        type=int,
        default=10,
        help='Number of samples to visualize'
    )
    
    args = parser.parse_args()
    
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        num_visualizations=args.num_vis,
    )
