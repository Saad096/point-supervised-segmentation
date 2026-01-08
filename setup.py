#!/usr/bin/env python3
"""
Setup Script for Point-Supervised Segmentation Project
Handles complete data preparation pipeline:
1. Create directory structure
2. Copy data from archive to proper locations
3. Verify data integrity
4. Generate train/val/test splits

Usage:
    python setup.py           # Interactive mode
    python setup.py --force   # Force re-copy data
    python setup.py --skip    # Skip if data exists
"""

import os
import sys
import shutil
from pathlib import Path
import json
import argparse

# Project root
PROJECT_ROOT = Path(__file__).parent

# Parse arguments
parser = argparse.ArgumentParser(description='Setup data for training')
parser.add_argument('--force', action='store_true', help='Force re-copy data')
parser.add_argument('--skip', action='store_true', help='Skip if data exists')
ARGS, _ = parser.parse_known_args()

def print_header(text):
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def print_step(step_num, text):
    """Print step information."""
    print(f"[Step {step_num}] {text}")

def print_success(text):
    """Print success message."""
    print(f"  [OK] {text}")

def print_error(text):
    """Print error message."""
    print(f"  [ERROR] {text}")

def create_directory_structure():
    """Create all required directories."""
    print_step(1, "Creating directory structure...")

    directories = [
        "data/raw/images",
        "data/raw/masks",
        "data/processed",
        "experiments/baseline",
        "experiments/optimized",
        "experiments/exp1_points_50",
        "experiments/exp1_points_200",
        "experiments/exp2_no_augmentation",
        "experiments/exp2_with_augmentation",
        "images/metrics",
        "images/samples",
        "mlflow_tracking",
    ]

    for dir_path in directories:
        full_path = PROJECT_ROOT / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print_success(f"Created: {dir_path}")

    return True

def copy_data_from_archive():
    """Copy images and masks from archive to data/raw."""
    print_step(2, "Copying data from archive...")

    archive_dir = PROJECT_ROOT / "archive"
    images_dst = PROJECT_ROOT / "data/raw/images"
    masks_dst = PROJECT_ROOT / "data/raw/masks"

    # Check if archive exists
    if not archive_dir.exists():
        print_error(f"Archive directory not found: {archive_dir}")
        print("  Please ensure the archive folder contains the DeepGlobe dataset")
        return False

    # Check archive subdirectories
    train_dir = archive_dir / "train"
    if not train_dir.exists():
        print_error(f"Training data not found: {train_dir}")
        return False

    # Count existing files
    existing_images = len(list(images_dst.glob("*.jpg")))
    existing_masks = len(list(masks_dst.glob("*.png")))

    if existing_images > 0 and existing_masks > 0:
        print_success(f"Data already exists: {existing_images} images, {existing_masks} masks")
        if ARGS.skip:
            print("  Skipping (--skip flag)")
            return True
        elif not ARGS.force:
            try:
                response = input("  Do you want to re-copy? (y/N): ").strip().lower()
                if response != 'y':
                    return True
            except EOFError:
                # Non-interactive mode, skip by default
                return True

    # Copy satellite images (*_sat.jpg)
    print("  Copying satellite images...")
    sat_files = list(train_dir.glob("*_sat.jpg"))
    copied_images = 0
    for src_file in sat_files:
        dst_file = images_dst / src_file.name
        shutil.copy2(src_file, dst_file)
        copied_images += 1
    print_success(f"Copied {copied_images} satellite images")

    # Copy mask files (*_mask.png)
    print("  Copying segmentation masks...")
    mask_files = list(train_dir.glob("*_mask.png"))
    copied_masks = 0
    for src_file in mask_files:
        dst_file = masks_dst / src_file.name
        shutil.copy2(src_file, dst_file)
        copied_masks += 1
    print_success(f"Copied {copied_masks} segmentation masks")

    return copied_images > 0 and copied_masks > 0

def verify_data_integrity():
    """Verify that images and masks are properly matched."""
    print_step(3, "Verifying data integrity...")

    images_dir = PROJECT_ROOT / "data/raw/images"
    masks_dir = PROJECT_ROOT / "data/raw/masks"

    # Get all files
    image_files = sorted(list(images_dir.glob("*_sat.jpg")))
    mask_files = sorted(list(masks_dir.glob("*_mask.png")))

    print(f"  Found {len(image_files)} images")
    print(f"  Found {len(mask_files)} masks")

    # Check matching
    matched = 0
    unmatched_images = []

    for img_path in image_files:
        # Extract ID: 123456_sat.jpg -> 123456
        img_id = img_path.stem.replace("_sat", "")
        mask_name = f"{img_id}_mask.png"
        mask_path = masks_dir / mask_name

        if mask_path.exists():
            matched += 1
        else:
            unmatched_images.append(img_path.name)

    if unmatched_images:
        print_error(f"Found {len(unmatched_images)} images without matching masks")
        for img in unmatched_images[:5]:
            print(f"    - {img}")
        return False

    print_success(f"All {matched} image-mask pairs verified")
    return True

def create_data_splits():
    """Create train/val/test splits and save to JSON."""
    print_step(4, "Creating train/val/test splits...")

    images_dir = PROJECT_ROOT / "data/raw/images"
    split_file = PROJECT_ROOT / "data/raw/split.json"

    # Get all image files
    image_files = sorted(list(images_dir.glob("*_sat.jpg")))
    n_samples = len(image_files)

    if n_samples == 0:
        print_error("No images found to split")
        return False

    # Create indices
    indices = list(range(n_samples))

    # Shuffle with fixed seed for reproducibility
    import random
    random.seed(42)
    random.shuffle(indices)

    # Split: 70% train, 15% val, 15% test
    train_end = int(0.7 * n_samples)
    val_end = int(0.85 * n_samples)

    splits = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:]
    }

    # Save splits
    with open(split_file, 'w') as f:
        json.dump(splits, f, indent=2)

    print_success(f"Train: {len(splits['train'])} samples")
    print_success(f"Val: {len(splits['val'])} samples")
    print_success(f"Test: {len(splits['test'])} samples")
    print_success(f"Splits saved to: {split_file}")

    return True

def print_summary():
    """Print final summary."""
    print_header("Setup Complete!")

    # Count files
    images = len(list((PROJECT_ROOT / "data/raw/images").glob("*.jpg")))
    masks = len(list((PROJECT_ROOT / "data/raw/masks").glob("*.png")))

    print("Data Summary:")
    print(f"  - Images: {images}")
    print(f"  - Masks: {masks}")
    print(f"  - Location: data/raw/")

    print("\nNext Steps:")
    print("  1. Train baseline model:")
    print("     python main.py --mode train --experiment baseline")
    print("")
    print("  2. Run all experiments:")
    print("     python main.py --mode train --experiment both")
    print("")
    print("  3. View MLflow dashboard:")
    print("     mlflow ui --backend-store-uri mlflow_tracking")
    print("     Open: http://localhost:5000")
    print("")

def main():
    """Main setup function."""
    print_header("Point-Supervised Segmentation - Setup")

    # Step 1: Create directories
    if not create_directory_structure():
        print_error("Failed to create directory structure")
        sys.exit(1)

    # Step 2: Copy data from archive
    if not copy_data_from_archive():
        print_error("Failed to copy data from archive")
        sys.exit(1)

    # Step 3: Verify data
    if not verify_data_integrity():
        print_error("Data verification failed")
        sys.exit(1)

    # Step 4: Create splits
    if not create_data_splits():
        print_error("Failed to create data splits")
        sys.exit(1)

    # Print summary
    print_summary()

if __name__ == "__main__":
    main()
