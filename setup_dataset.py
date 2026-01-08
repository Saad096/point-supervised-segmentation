"""
Dataset Setup and Verification Script
Helps users set up the DeepGlobe dataset and verify it's correctly formatted
"""

import os
import sys
from pathlib import Path
import json
from typing import Tuple
import cv2
import numpy as np

def check_directory_structure(data_dir: Path) -> Tuple[bool, str]:
    """
    Check if the dataset directory has the correct structure.
    
    Returns:
        tuple: (is_valid, message)
    """
    # Check if data_dir exists
    if not data_dir.exists():
        return False, f"Directory not found: {data_dir}"
    
    # Check for required subdirectories
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    
    if not images_dir.exists():
        return False, f"Images directory not found: {images_dir}"
    
    if not masks_dir.exists():
        return False, f"Masks directory not found: {masks_dir}"
    
    # Check for image files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    mask_files = list(masks_dir.glob("*.png")) + list(masks_dir.glob("*.jpg"))
    
    if len(image_files) == 0:
        return False, f"No image files found in {images_dir}"
    
    if len(mask_files) == 0:
        return False, f"No mask files found in {masks_dir}"
    
    # Check if counts match
    if abs(len(image_files) - len(mask_files)) > 5:
        return False, f"Mismatch: {len(image_files)} images but {len(mask_files)} masks"
    
    return True, f"Found {len(image_files)} images and {len(mask_files)} masks"


def verify_dataset_integrity(data_dir: Path, num_samples: int = 10) -> dict:
    """
    Verify dataset integrity by checking a few samples.
    
    Args:
        data_dir: Path to dataset
        num_samples: Number of samples to verify
        
    Returns:
        dict: Verification results
    """
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    results = {
        'valid_samples': 0,
        'invalid_samples': 0,
        'issues': [],
        'image_sizes': [],
        'mask_sizes': [],
    }
    
    for i, img_path in enumerate(image_files[:num_samples]):
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                results['issues'].append(f"Failed to load image: {img_path.name}")
                results['invalid_samples'] += 1
                continue
            
            # Try to find corresponding mask
            mask_name = img_path.stem + "_mask" + img_path.suffix
            mask_path = masks_dir / mask_name
            
            if not mask_path.exists():
                mask_name = img_path.stem + img_path.suffix
                mask_path = masks_dir / mask_name
            
            if not mask_path.exists():
                results['issues'].append(f"No mask found for image: {img_path.name}")
                results['invalid_samples'] += 1
                continue
            
            # Load mask
            mask = cv2.imread(str(mask_path))
            if mask is None:
                results['issues'].append(f"Failed to load mask: {mask_path.name}")
                results['invalid_samples'] += 1
                continue
            
            # Check dimensions
            if img.shape[:2] != mask.shape[:2]:
                results['issues'].append(
                    f"Size mismatch: {img_path.name} {img.shape[:2]} vs "
                    f"{mask_path.name} {mask.shape[:2]}"
                )
                results['invalid_samples'] += 1
                continue
            
            results['valid_samples'] += 1
            results['image_sizes'].append(img.shape)
            results['mask_sizes'].append(mask.shape)
            
        except Exception as e:
            results['issues'].append(f"Error processing {img_path.name}: {e}")
            results['invalid_samples'] += 1
    
    return results


def generate_dataset_statistics(data_dir: Path) -> dict:
    """
    Generate comprehensive dataset statistics.
    
    Args:
        data_dir: Path to dataset
        
    Returns:
        dict: Dataset statistics
    """
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    mask_files = sorted(list(masks_dir.glob("*.png")) + list(masks_dir.glob("*.jpg")))
    
    stats = {
        'total_images': len(image_files),
        'total_masks': len(mask_files),
        'dataset_size_mb': 0,
        'average_image_size': None,
        'unique_image_sizes': set(),
    }
    
    # Calculate total size
    for img_path in image_files:
        stats['dataset_size_mb'] += img_path.stat().st_size / (1024 * 1024)
    
    for mask_path in mask_files:
        stats['dataset_size_mb'] += mask_path.stat().st_size / (1024 * 1024)
    
    # Sample images for size statistics
    sample_sizes = []
    for img_path in image_files[:50]:  # Sample first 50
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                sample_sizes.append(img.shape[:2])
                stats['unique_image_sizes'].add(img.shape[:2])
        except:
            continue
    
    if sample_sizes:
        avg_h = sum(h for h, w in sample_sizes) / len(sample_sizes)
        avg_w = sum(w for h, w in sample_sizes) / len(sample_sizes)
        stats['average_image_size'] = (int(avg_h), int(avg_w))
    
    stats['unique_image_sizes'] = [size for size in stats['unique_image_sizes']]
    
    return stats


def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("DeepGlobe Dataset Setup and Verification")
    print("="*80 + "\n")
    
    # Get data directory
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        data_dir = Path("data/raw")
    
    print(f"Checking dataset at: {data_dir.absolute()}\n")
    
    # Step 1: Check directory structure
    print("Step 1: Verifying directory structure...")
    is_valid, message = check_directory_structure(data_dir)
    
    if is_valid:
        print(f"✓ {message}")
    else:
        print(f"✗ {message}")
        print("\nExpected structure:")
        print("  data/raw/")
        print("    ├── images/")
        print("    │   ├── 001_sat.jpg")
        print("    │   └── ...")
        print("    └── masks/")
        print("        ├── 001_mask.png")
        print("        └── ...")
        print("\nPlease download the DeepGlobe dataset from:")
        print("https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset")
        sys.exit(1)
    
    # Step 2: Verify integrity
    print("\nStep 2: Verifying dataset integrity (checking 10 samples)...")
    results = verify_dataset_integrity(data_dir, num_samples=10)
    
    print(f"  Valid samples: {results['valid_samples']}")
    print(f"  Invalid samples: {results['invalid_samples']}")
    
    if results['issues']:
        print("\n  Issues found:")
        for issue in results['issues'][:5]:  # Show first 5 issues
            print(f"    - {issue}")
        if len(results['issues']) > 5:
            print(f"    ... and {len(results['issues']) - 5} more")
    
    if results['valid_samples'] == 0:
        print("\n✗ No valid samples found. Please check your dataset.")
        sys.exit(1)
    
    print("✓ Dataset integrity verified")
    
    # Step 3: Generate statistics
    print("\nStep 3: Generating dataset statistics...")
    stats = generate_dataset_statistics(data_dir)
    
    print(f"  Total images: {stats['total_images']}")
    print(f"  Total masks: {stats['total_masks']}")
    print(f"  Dataset size: {stats['dataset_size_mb']:.2f} MB")
    if stats['average_image_size']:
        print(f"  Average image size: {stats['average_image_size']}")
    print(f"  Unique image sizes: {len(stats['unique_image_sizes'])}")
    
    # Step 4: Create split file if it doesn't exist
    split_file = data_dir / "split.json"
    if not split_file.exists():
        print("\nStep 4: Creating train/val/test split...")
        from sklearn.model_selection import train_test_split
        
        n_samples = stats['total_images']
        indices = list(range(n_samples))
        
        # 70% train, 15% val, 15% test
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
        
        splits = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
        
        with open(split_file, 'w') as f:
            json.dump(splits, f, indent=2)
        
        print(f"  Train samples: {len(train_idx)}")
        print(f"  Val samples: {len(val_idx)}")
        print(f"  Test samples: {len(test_idx)}")
        print(f"  Split file saved to: {split_file}")
    else:
        print("\nStep 4: Split file already exists")
        with open(split_file, 'r') as f:
            splits = json.load(f)
        print(f"  Train samples: {len(splits['train'])}")
        print(f"  Val samples: {len(splits['val'])}")
        print(f"  Test samples: {len(splits['test'])}")
    
    # Final summary
    print("\n" + "="*80)
    print("✓ Dataset setup complete!")
    print("="*80)
    print("\nYou can now run experiments:")
    print(f"  python run_experiments.py --data_dir {data_dir} --experiment baseline")
    print("\nOr run all experiments:")
    print(f"  python run_experiments.py --data_dir {data_dir} --experiment all")
    print()


if __name__ == "__main__":
    main()
