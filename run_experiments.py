"""
Main Experiment Runner for Point-Supervised Segmentation
Orchestrates different experiments and ablation studies
"""

import os
import sys
import torch
import torch.optim as optim
from pathlib import Path
import argparse
import mlflow
from datetime import datetime

# Add project paths
sys.path.append(str(Path(__file__).parent / 'configs'))
sys.path.append(str(Path(__file__).parent / 'src'))

from config import *
from model import get_model
from loss import get_loss_function
from data_loader import get_dataloaders
from point_sampling import PointSampler, AdaptivePointSampler
from train import Trainer


def run_experiment(
    experiment_name: str,
    experiment_config: dict,
    data_dir: str,
    device: str = None,
):
    """
    Run a single experiment.
    
    Args:
        experiment_name: Name of the experiment
        experiment_config: Experiment configuration dictionary
        data_dir: Path to dataset
        device: Device to run on (auto-detect if None)
    """
    print(f"\n{'='*100}")
    print(f"Running Experiment: {experiment_name}")
    print(f"{'='*100}\n")
    
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = EXPERIMENTS_DIR / experiment_name / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config
    config = {
        **DATASET_CONFIG,
        **MODEL_CONFIG,
        **TRAIN_CONFIG,
        **LOSS_CONFIG,
        **EVAL_CONFIG,
        'experiment_name': experiment_name,
        'device': device,
    }
    
    # Apply experiment-specific config
    config.update(experiment_config)
    
    # Create point sampler
    num_points = experiment_config.get('num_points', 100)
    sampling_strategy = experiment_config.get('sampling_strategy', 'random')
    
    point_sampler = PointSampler(
        num_points_per_class=num_points,
        sampling_strategy=sampling_strategy,
        min_distance=POINT_SAMPLING_CONFIG['min_distance'],
        ignore_classes=[6],  # Ignore 'unknown' class
        seed=42,
    )
    
    print(f"Point Sampler Configuration:")
    print(f"  Points per class: {num_points}")
    print(f"  Sampling strategy: {sampling_strategy}")
    
    # Create data loaders
    print("\nLoading dataset...")
    use_augmentation = experiment_config.get('augmentation', True)
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        num_workers=config.get('num_workers', 4),
        point_sampler=point_sampler,
        use_augmentation=use_augmentation,
        pin_memory=config.get('pin_memory', True),
    )
    
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
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
    print("\nCreating loss function...")
    loss_fn = get_loss_function(config)
    print(f"  Loss type: {config['type']}")
    
    # Create optimizer
    print("\nCreating optimizer...")
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5),
        )
    elif config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5),
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=0.9,
            weight_decay=config.get('weight_decay', 1e-5),
        )
    
    print(f"  Optimizer: {config['optimizer']}")
    print(f"  Learning rate: {config['learning_rate']}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        experiment_dir=experiment_dir,
        use_mlflow=True,
    )
    
    # Start training
    print("\n" + "="*100)
    print("Starting Training")
    print("="*100 + "\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_results()
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    
    print(f"\n{'='*100}")
    print(f"Experiment Complete: {experiment_name}")
    print(f"Results saved to: {experiment_dir}")
    print(f"{'='*100}\n")
    
    return experiment_dir, trainer.best_val_iou


def run_baseline_experiment(data_dir: str, device: str = None):
    """
    Run baseline experiment with minimal points and no augmentation.
    """
    config = EXPERIMENTS['baseline'].copy()
    return run_experiment('baseline', config, data_dir, device)


def run_optimized_experiment(data_dir: str, device: str = None):
    """
    Run optimized experiment with more points and augmentation.
    """
    config = EXPERIMENTS['optimized'].copy()
    return run_experiment('optimized', config, data_dir, device)


def run_ablation_points(data_dir: str, device: str = None):
    """
    Run ablation study on point density.
    """
    print(f"\n{'#'*100}")
    print(f"# ABLATION STUDY: Effect of Point Density")
    print(f"{'#'*100}\n")
    
    results = {}
    point_variants = EXPERIMENTS['ablation_points']['num_points_variants']
    
    for num_points in point_variants:
        experiment_name = f'ablation_points/points_{num_points}'
        config = {
            'num_points': num_points,
            'augmentation': True,
            'num_epochs': EXPERIMENTS['ablation_points']['num_epochs'],
        }
        
        exp_dir, best_iou = run_experiment(experiment_name, config, data_dir, device)
        results[num_points] = {
            'experiment_dir': exp_dir,
            'best_iou': best_iou,
        }
    
    # Print summary
    print(f"\n{'='*100}")
    print("Ablation Study Results: Point Density")
    print(f"{'='*100}")
    for num_points, res in results.items():
        print(f"  {num_points} points per class: IoU = {res['best_iou']:.4f}")
    print(f"{'='*100}\n")
    
    return results


def run_ablation_augmentation(data_dir: str, device: str = None):
    """
    Run ablation study on data augmentation.
    """
    print(f"\n{'#'*100}")
    print(f"# ABLATION STUDY: Effect of Data Augmentation")
    print(f"{'#'*100}\n")
    
    results = {}
    augmentation_variants = EXPERIMENTS['ablation_augmentation']['augmentation_variants']
    
    for use_aug in augmentation_variants:
        experiment_name = f'ablation_augmentation/aug_{use_aug}'
        config = {
            'num_points': EXPERIMENTS['ablation_augmentation']['num_points'],
            'augmentation': use_aug,
            'num_epochs': EXPERIMENTS['ablation_augmentation']['num_epochs'],
        }
        
        exp_dir, best_iou = run_experiment(experiment_name, config, data_dir, device)
        results[use_aug] = {
            'experiment_dir': exp_dir,
            'best_iou': best_iou,
        }
    
    # Print summary
    print(f"\n{'='*100}")
    print("Ablation Study Results: Data Augmentation")
    print(f"{'='*100}")
    for use_aug, res in results.items():
        aug_str = "With Augmentation" if use_aug else "Without Augmentation"
        print(f"  {aug_str}: IoU = {res['best_iou']:.4f}")
    print(f"{'='*100}\n")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Point-Supervised Segmentation Experiments'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/raw',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['baseline', 'optimized', 'ablation_points', 'ablation_aug', 'all'],
        default='all',
        help='Which experiment to run'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default=None,
        help='Device to run on (auto-detect if not specified)'
    )
    
    args = parser.parse_args()
    
    # Setup MLflow
    mlflow.set_tracking_uri(str(MLFLOW_DIR))
    mlflow.set_experiment(MLFLOW_CONFIG['experiment_name'])
    
    print(f"\n{'#'*100}")
    print(f"# Point-Supervised Semantic Segmentation")
    print(f"# Remote Sensing Image Segmentation with Sparse Point Annotations")
    print(f"{'#'*100}\n")
    
    print(f"Configuration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Experiment: {args.experiment}")
    print(f"  Device: {args.device or 'auto-detect'}")
    print(f"  MLflow tracking: {MLFLOW_DIR}")
    
    # Run experiments
    if args.experiment == 'baseline':
        run_baseline_experiment(args.data_dir, args.device)
    
    elif args.experiment == 'optimized':
        run_optimized_experiment(args.data_dir, args.device)
    
    elif args.experiment == 'ablation_points':
        run_ablation_points(args.data_dir, args.device)
    
    elif args.experiment == 'ablation_aug':
        run_ablation_augmentation(args.data_dir, args.device)
    
    elif args.experiment == 'all':
        # Run all experiments
        print("\n" + "="*100)
        print("Running ALL experiments")
        print("="*100 + "\n")
        
        run_baseline_experiment(args.data_dir, args.device)
        run_optimized_experiment(args.data_dir, args.device)
        run_ablation_points(args.data_dir, args.device)
        run_ablation_augmentation(args.data_dir, args.device)
    
    print(f"\n{'#'*100}")
    print(f"# All Experiments Complete!")
    print(f"{'#'*100}\n")


if __name__ == "__main__":
    main()
