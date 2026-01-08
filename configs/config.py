"""
Configuration file for Point-Supervised Segmentation Experiments
"""

import torch
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent  # Go up from configs/ to project root
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports"
IMAGES_DIR = PROJECT_ROOT / "images"
MLFLOW_DIR = PROJECT_ROOT / "mlflow_tracking"

# Dataset Configuration
DATASET_CONFIG = {
    "name": "DeepGlobe Land Cover",
    "num_classes": 7,  # urban, agriculture, rangeland, forest, water, barren, unknown
    "class_names": [
        "urban_land",
        "agriculture_land", 
        "rangeland",
        "forest_land",
        "water",
        "barren_land",
        "unknown"
    ],
    "class_colors": [
        (0, 255, 255),    # urban - cyan
        (255, 255, 0),    # agriculture - yellow
        (255, 0, 255),    # rangeland - magenta
        (0, 255, 0),      # forest - green
        (0, 0, 255),      # water - blue
        (255, 255, 255),  # barren - white
        (0, 0, 0)         # unknown - black
    ],
    "image_size": (512, 512),
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15
}

# Point Sampling Configuration
POINT_SAMPLING_CONFIG = {
    "num_points_per_class": [50, 100, 200, 500],  # For ablation study
    "sampling_strategy": "random",  # random, stratified, grid
    "min_distance": 10,  # Minimum distance between points (pixels)
}

# Model Configuration
MODEL_CONFIG = {
    "architecture": "unet",
    "encoder": "resnet34",
    "encoder_weights": "imagenet",
    "in_channels": 3,
    "classes": 7,
    "activation": None,
}

# Training Configuration
TRAIN_CONFIG = {
    "batch_size": 8,
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "optimizer": "adam",
    "scheduler": "reduce_on_plateau",
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "early_stopping_patience": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
    "pin_memory": True,
    "mixed_precision": True,
}

# Partial Cross Entropy Loss Configuration
LOSS_CONFIG = {
    "type": "partial_ce",
    "ignore_index": -1,
    "weight": None,  # Class weights (optional)
    "lambda_point": 1.0,  # Weight for point supervision
    "lambda_consistency": 0.1,  # Weight for consistency regularization
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    "train": {
        "horizontal_flip": 0.5,
        "vertical_flip": 0.5,
        "rotate": 45,
        "random_brightness_contrast": 0.2,
        "random_gamma": (80, 120),
        "gaussian_noise": 0.01,
        "elastic_transform": True,
    },
    "val_test": {
        # No augmentation for validation/test
    }
}

# Evaluation Configuration
EVAL_CONFIG = {
    "metrics": ["accuracy", "iou", "f1", "precision", "recall"],
    "save_predictions": True,
    "num_visualization_samples": 10,
}

# MLflow Configuration
MLFLOW_CONFIG = {
    "tracking_uri": str(MLFLOW_DIR),
    "experiment_name": "point_supervised_segmentation",
    "artifact_location": str(MLFLOW_DIR / "artifacts"),
}

# Experiment Configurations
EXPERIMENTS = {
    "baseline": {
        "name": "Baseline - 50 points per class",
        "num_points": 50,
        "augmentation": False,
        "num_epochs": 30,
    },
    "optimized": {
        "name": "Optimized - 200 points + augmentation",
        "num_points": 200,
        "augmentation": True,
        "num_epochs": 50,
    },
    "ablation_points": {
        "name": "Ablation Study - Point Density",
        "num_points_variants": [50, 100, 200, 500],
        "augmentation": True,
        "num_epochs": 40,
    },
    "ablation_augmentation": {
        "name": "Ablation Study - Data Augmentation",
        "num_points": 200,
        "augmentation_variants": [False, True],
        "num_epochs": 40,
    }
}
