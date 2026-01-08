#!/usr/bin/env python3
"""
Main Entry Point for Point-Supervised Segmentation
Simple interface for training and evaluation

Usage:
    # Train with default settings
    python main.py --mode train

    # Train with specific experiment
    python main.py --mode train --experiment baseline

    # Run both experiments for comparison
    python main.py --mode train --experiment both

    # Evaluate a trained model
    python main.py --mode evaluate --checkpoint experiments/baseline/best_model.pth

    # Run inference on a single image
    python main.py --mode inference --checkpoint experiments/baseline/best_model.pth --image path/to/image.jpg
"""

import os
import sys
import torch
import torch.optim as optim
from pathlib import Path
import argparse
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add project paths
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / 'configs'))
sys.path.append(str(PROJECT_ROOT / 'src'))

from config import *
from model import get_model
from loss import get_loss_function
from data_loader import get_dataloaders, DeepGlobeDataset, get_transforms
from point_sampling import PointSampler
from train import Trainer
from evaluate import ModelEvaluator


def train_model(
    experiment_name: str,
    num_points: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    use_augmentation: bool,
    data_dir: str,
    device: str,
    sampling_strategy: str = 'random',
):
    """
    Train a point-supervised segmentation model.

    Args:
        experiment_name: Name for this experiment
        num_points: Number of point labels per class
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        use_augmentation: Whether to use data augmentation
        data_dir: Path to data directory
        device: Device to train on
        sampling_strategy: Point sampling strategy

    Returns:
        tuple: (experiment_dir, best_iou)
    """
    print(f"\n{'='*80}")
    print(f"Training: {experiment_name}")
    print(f"{'='*80}")

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = EXPERIMENTS_DIR / experiment_name / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    config = {
        'experiment_name': experiment_name,
        'num_classes': DATASET_CONFIG['num_classes'],
        'class_names': DATASET_CONFIG['class_names'],
        'class_colors': DATASET_CONFIG['class_colors'],
        'image_size': DATASET_CONFIG['image_size'],
        'architecture': MODEL_CONFIG['architecture'],
        'encoder': MODEL_CONFIG['encoder'],
        'encoder_weights': MODEL_CONFIG['encoder_weights'],
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'weight_decay': 1e-5,
        'optimizer': 'adam',
        'scheduler': 'reduce_on_plateau',
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
        'early_stopping_patience': 15,
        'num_workers': 4,
        'mixed_precision': torch.cuda.is_available(),
        'type': 'partial_ce',
        'ignore_index': -1,
        'lambda_point': 1.0,
        'num_points': num_points,
        'sampling_strategy': sampling_strategy,
        'use_augmentation': use_augmentation,
        'device': device,
    }

    print(f"\nConfiguration:")
    print(f"  Points per class: {num_points}")
    print(f"  Sampling strategy: {sampling_strategy}")
    print(f"  Data augmentation: {use_augmentation}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Device: {device}")

    # Create point sampler
    point_sampler = PointSampler(
        num_points_per_class=num_points,
        sampling_strategy=sampling_strategy,
        min_distance=10,
        ignore_classes=[6],  # Ignore 'unknown' class
        seed=42,
    )

    # Create data loaders
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=config['image_size'],
        num_workers=config['num_workers'],
        point_sampler=point_sampler,
        use_augmentation=use_augmentation,
        pin_memory=True,
    )

    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")

    # Create model
    print("\nCreating model...")
    model = get_model(
        architecture=config['architecture'],
        encoder_name=config['encoder'],
        encoder_weights=config['encoder_weights'],
        num_classes=config['num_classes'],
        device=device,
    )

    # Create loss function
    loss_fn = get_loss_function(config)

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config['weight_decay'],
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        experiment_dir=experiment_dir,
        use_mlflow=False,  # Disable MLflow for simplicity
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Copy training curves to images directory
    import shutil
    curves_src = experiment_dir / 'training_curves.png'
    if curves_src.exists():
        curves_dst = IMAGES_DIR / 'metrics' / f'{experiment_name}_training_curves.png'
        curves_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(curves_src, curves_dst)
        print(f"Training curves saved to: {curves_dst}")

    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"Best IoU: {trainer.best_val_iou:.4f}")
    print(f"Results saved to: {experiment_dir}")
    print(f"{'='*80}\n")

    return experiment_dir, trainer.best_val_iou


def evaluate_model(checkpoint_path: str, data_dir: str, device: str):
    """Evaluate a trained model."""
    print(f"\n{'='*80}")
    print(f"Evaluating model: {checkpoint_path}")
    print(f"{'='*80}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Create output directory
    output_dir = Path(checkpoint_path).parent / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    model = get_model(
        architecture=config['architecture'],
        encoder_name=config['encoder'],
        encoder_weights=None,
        num_classes=config['num_classes'],
        device=device,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create test loader
    transform = get_transforms('test', config['image_size'])
    test_dataset = DeepGlobeDataset(
        data_dir=data_dir,
        split='test',
        image_size=config['image_size'],
        point_sampler=None,
        transform=transform,
        use_point_labels=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
    )

    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=config['class_names'],
        class_colors=config.get('class_colors', DATASET_CONFIG['class_colors']),
        output_dir=output_dir,
    )

    # Evaluate
    metrics = evaluator.evaluate()
    evaluator.visualize_predictions(num_samples=10)
    evaluator.generate_report()

    # Copy visualizations to images directory
    import shutil
    vis_src = output_dir / 'confusion_matrix.png'
    if vis_src.exists():
        vis_dst = IMAGES_DIR / 'metrics' / f'{Path(checkpoint_path).parent.parent.name}_confusion_matrix.png'
        vis_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(vis_src, vis_dst)

    return metrics


def run_experiments(data_dir: str, device: str):
    """
    Run two experiments to compare performance factors.

    Experiment 1: Effect of Point Density
        - Compare 50 vs 200 points per class

    Experiment 2: Effect of Data Augmentation
        - Compare with/without augmentation
    """
    results = {}

    # ========================================
    # Experiment 1: Point Density Comparison
    # ========================================
    print("\n" + "#"*80)
    print("# EXPERIMENT 1: Effect of Point Density")
    print("#"*80)

    # Low point density (50 points)
    exp_dir_50, iou_50 = train_model(
        experiment_name='exp1_points_50',
        num_points=50,
        num_epochs=30,
        batch_size=4,
        learning_rate=1e-4,
        use_augmentation=True,
        data_dir=data_dir,
        device=device,
    )
    results['exp1_points_50'] = {'iou': iou_50, 'dir': str(exp_dir_50)}

    # High point density (200 points)
    exp_dir_200, iou_200 = train_model(
        experiment_name='exp1_points_200',
        num_points=200,
        num_epochs=30,
        batch_size=4,
        learning_rate=1e-4,
        use_augmentation=True,
        data_dir=data_dir,
        device=device,
    )
    results['exp1_points_200'] = {'iou': iou_200, 'dir': str(exp_dir_200)}

    # ========================================
    # Experiment 2: Data Augmentation Effect
    # ========================================
    print("\n" + "#"*80)
    print("# EXPERIMENT 2: Effect of Data Augmentation")
    print("#"*80)

    # Without augmentation
    exp_dir_no_aug, iou_no_aug = train_model(
        experiment_name='exp2_no_augmentation',
        num_points=100,
        num_epochs=30,
        batch_size=4,
        learning_rate=1e-4,
        use_augmentation=False,
        data_dir=data_dir,
        device=device,
    )
    results['exp2_no_augmentation'] = {'iou': iou_no_aug, 'dir': str(exp_dir_no_aug)}

    # With augmentation
    exp_dir_aug, iou_aug = train_model(
        experiment_name='exp2_with_augmentation',
        num_points=100,
        num_epochs=30,
        batch_size=4,
        learning_rate=1e-4,
        use_augmentation=True,
        data_dir=data_dir,
        device=device,
    )
    results['exp2_with_augmentation'] = {'iou': iou_aug, 'dir': str(exp_dir_aug)}

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    print("\nExperiment 1: Effect of Point Density")
    print(f"  50 points per class:  IoU = {iou_50:.4f}")
    print(f"  200 points per class: IoU = {iou_200:.4f}")
    print(f"  Improvement: {(iou_200 - iou_50)*100:.2f}%")

    print("\nExperiment 2: Effect of Data Augmentation")
    print(f"  Without augmentation: IoU = {iou_no_aug:.4f}")
    print(f"  With augmentation:    IoU = {iou_aug:.4f}")
    print(f"  Improvement: {(iou_aug - iou_no_aug)*100:.2f}%")

    print("="*80)

    # Save results
    results_file = EXPERIMENTS_DIR / 'experiment_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Point-Supervised Segmentation for Remote Sensing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train --experiment baseline
  python main.py --mode train --experiment both
  python main.py --mode evaluate --checkpoint experiments/baseline/best_model.pth
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'inference', 'experiments'],
        default='train',
        help='Mode: train, evaluate, inference, or experiments'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['baseline', 'optimized', 'both'],
        default='baseline',
        help='Experiment to run (for train mode)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint (for evaluate mode)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/raw',
        help='Path to data directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/cpu, auto-detect if not specified)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--num_points',
        type=int,
        default=100,
        help='Number of point labels per class'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=3,
        help='Number of samples for inference'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Output directory for inference results'
    )

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("\n" + "#"*80)
    print("# Point-Supervised Segmentation for Remote Sensing Images")
    print("# Using Partial Cross-Entropy Loss with Point Annotations")
    print("#"*80)
    print(f"\nDevice: {device}")
    print(f"Data directory: {args.data_dir}")

    if args.mode == 'train':
        if args.experiment == 'baseline':
            train_model(
                experiment_name='baseline',
                num_points=50,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                use_augmentation=False,
                data_dir=args.data_dir,
                device=device,
            )
        elif args.experiment == 'optimized':
            train_model(
                experiment_name='optimized',
                num_points=200,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                use_augmentation=True,
                data_dir=args.data_dir,
                device=device,
            )
        elif args.experiment == 'both':
            run_experiments(args.data_dir, device)

    elif args.mode == 'evaluate':
        if args.checkpoint is None:
            print("Error: --checkpoint required for evaluate mode")
            sys.exit(1)
        evaluate_model(args.checkpoint, args.data_dir, device)

    elif args.mode == 'inference':
        if args.checkpoint is None:
            print("Error: --checkpoint required for inference mode")
            sys.exit(1)
        # Import and run inference
        import subprocess
        cmd = [
            sys.executable, 'inference.py',
            '--checkpoint', args.checkpoint,
            '--data_dir', args.data_dir,
            '--output_dir', args.output_dir,
            '--num_samples', str(args.num_samples),
            '--device', device
        ]
        subprocess.run(cmd)

    elif args.mode == 'experiments':
        run_experiments(args.data_dir, device)


if __name__ == '__main__':
    main()
