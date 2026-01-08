#!/usr/bin/env python3
"""
Inference Script for Point-Supervised Segmentation
Run predictions on sample images and save visualizations

Usage:
    python inference.py --checkpoint experiments/baseline/[timestamp]/best_model.pth
    python inference.py --checkpoint experiments/baseline/[timestamp]/best_model.pth --num_samples 5
"""

import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import random

# Add paths
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / 'src'))
sys.path.append(str(PROJECT_ROOT / 'configs'))

from model import get_model
from data_loader import get_transforms


# Class configuration
CLASS_NAMES = [
    "Urban", "Agriculture", "Rangeland",
    "Forest", "Water", "Barren", "Unknown"
]

CLASS_COLORS = [
    (0, 255, 255),    # Urban - Cyan
    (255, 255, 0),    # Agriculture - Yellow
    (255, 0, 255),    # Rangeland - Magenta
    (0, 255, 0),      # Forest - Green
    (0, 0, 255),      # Water - Blue
    (255, 255, 255),  # Barren - White
    (0, 0, 0)         # Unknown - Black
]


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = get_model(
        architecture=config['architecture'],
        encoder_name=config['encoder'],
        encoder_weights=None,
        num_classes=config['num_classes'],
        device=device,
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded (trained for {checkpoint['epoch']} epochs)")
    print(f"Best validation IoU: {checkpoint.get('best_val_iou', 'N/A')}")

    return model, config


def preprocess_image(image_path, image_size=(512, 512)):
    """Load and preprocess image for inference."""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Store original for visualization
    original = image.copy()

    # Resize
    image = cv2.resize(image, (image_size[1], image_size[0]))

    # Normalize
    transform = get_transforms('test', image_size)
    transformed = transform(image=image)
    tensor = transformed['image'].unsqueeze(0)

    return tensor, original, image


def mask_to_color(mask):
    """Convert class indices to RGB visualization."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, color in enumerate(CLASS_COLORS):
        colored[mask == class_idx] = color

    return colored


def create_legend():
    """Create a legend image for classes."""
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.axis('off')

    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        color_normalized = tuple(c/255 for c in color)
        ax.add_patch(plt.Rectangle((i*1.4, 0), 1, 0.5, facecolor=color_normalized, edgecolor='black'))
        ax.text(i*1.4 + 0.5, -0.3, name, ha='center', fontsize=8)

    ax.set_xlim(-0.1, len(CLASS_NAMES)*1.4)
    ax.set_ylim(-0.6, 0.6)

    return fig


@torch.no_grad()
def run_inference(model, image_tensor, device):
    """Run model inference."""
    image_tensor = image_tensor.to(device)
    output = model(image_tensor)
    prediction = output.argmax(dim=1).squeeze().cpu().numpy()
    return prediction


def visualize_prediction(original, resized, prediction, save_path, image_name):
    """Create and save visualization."""
    # Convert prediction to colored mask
    pred_colored = mask_to_color(prediction)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(resized)
    axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Prediction
    axes[1].imshow(pred_colored)
    axes[1].set_title('Segmentation Output', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Overlay
    overlay = cv2.addWeighted(resized, 0.6, pred_colored, 0.4, 0)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Add legend
    legend_text = " | ".join([f"{name}" for name in CLASS_NAMES[:6]])
    fig.text(0.5, 0.02, f"Classes: {legend_text}", ha='center', fontsize=9)

    plt.suptitle(f'Inference Result: {image_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    # Save
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Run inference on sample images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of sample images')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"\n{'='*60}")
    print("Point-Supervised Segmentation - Inference")
    print(f"{'='*60}")
    print(f"Device: {device}")

    # Load model
    model, config = load_model(args.checkpoint, device)
    image_size = config.get('image_size', (512, 512))

    # Get sample images
    images_dir = Path(args.data_dir) / 'images'
    image_files = list(images_dir.glob('*_sat.jpg'))

    if len(image_files) == 0:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    # Random sample
    random.seed(42)
    sample_images = random.sample(image_files, min(args.num_samples, len(image_files)))

    print(f"\nRunning inference on {len(sample_images)} images...")
    print(f"Output directory: {args.output_dir}\n")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    results = []
    for i, image_path in enumerate(sample_images):
        print(f"[{i+1}/{len(sample_images)}] Processing: {image_path.name}")

        # Preprocess
        tensor, original, resized = preprocess_image(image_path, image_size)

        # Inference
        prediction = run_inference(model, tensor, device)

        # Save visualization
        output_name = f"inference_result_{i+1}.png"
        output_path = output_dir / output_name
        visualize_prediction(original, resized, prediction, output_path, image_path.stem)

        results.append({
            'input': image_path.name,
            'output': output_name
        })

    # Summary
    print(f"\n{'='*60}")
    print("Inference Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for r in results:
        print(f"  - {r['output']} (from {r['input']})")

    return results


if __name__ == '__main__':
    main()
