"""
Data Loader for DeepGlobe Land Cover Classification Dataset
Handles loading, preprocessing, and point annotation sampling
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Callable, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import json

from point_sampling import PointSampler


class DeepGlobeDataset(Dataset):
    """
    DeepGlobe Land Cover Classification Dataset.
    
    This dataset contains satellite imagery with 7 land cover classes:
    - Urban land
    - Agriculture land
    - Rangeland
    - Forest land
    - Water
    - Barren land
    - Unknown
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (512, 512),
        point_sampler: Optional[PointSampler] = None,
        transform: Optional[Callable] = None,
        use_point_labels: bool = True,
        cache_data: bool = False,
    ):
        """
        Args:
            data_dir: Root directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size (H, W)
            point_sampler: Point sampler for generating sparse annotations
            transform: Albumentations transform pipeline
            use_point_labels: If True, use point labels; if False, use dense masks
            cache_data: If True, cache preprocessed data in memory
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.point_sampler = point_sampler
        self.transform = transform
        self.use_point_labels = use_point_labels
        self.cache_data = cache_data
        
        # Dataset configuration
        self.num_classes = 7
        self.class_names = [
            "urban_land", "agriculture_land", "rangeland",
            "forest_land", "water", "barren_land", "unknown"
        ]
        
        # RGB color mapping for masks
        self.color_map = {
            (0, 255, 255): 0,    # urban - cyan
            (255, 255, 0): 1,    # agriculture - yellow
            (255, 0, 255): 2,    # rangeland - magenta
            (0, 255, 0): 3,      # forest - green
            (0, 0, 255): 4,      # water - blue
            (255, 255, 255): 5,  # barren - white
            (0, 0, 0): 6         # unknown - black
        }
        
        # Load file paths
        self.image_paths, self.mask_paths = self._load_file_paths()
        
        # Cache for preprocessed data
        self.cache = {} if cache_data else None
        
        print(f"Loaded {len(self.image_paths)} samples for {split} split")
    
    def _load_file_paths(self) -> Tuple[List[Path], List[Path]]:
        """
        Load image and mask file paths from the dataset directory.
        """
        # DeepGlobe structure: data/raw/images/ and data/raw/masks/
        image_dir = self.data_dir / "images"
        mask_dir = self.data_dir / "masks"

        if not image_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(
                f"Dataset directories not found. Expected:\n"
                f"  {image_dir}\n"
                f"  {mask_dir}"
            )

        # Get all image files
        image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))

        # Match corresponding mask files
        image_paths = []
        mask_paths = []

        for img_path in image_files:
            # Handle DeepGlobe naming convention: 123456_sat.jpg -> 123456_mask.png
            if img_path.stem.endswith('_sat'):
                # Extract base ID (e.g., "123456" from "123456_sat")
                base_id = img_path.stem.replace('_sat', '')
                mask_name = f"{base_id}_mask.png"
            else:
                # Try different mask naming conventions
                mask_name = img_path.stem + "_mask.png"

            mask_path = mask_dir / mask_name

            if not mask_path.exists():
                # Fallback: try same name with png extension
                mask_name = img_path.stem + ".png"
                mask_path = mask_dir / mask_name

            if mask_path.exists():
                image_paths.append(img_path)
                mask_paths.append(mask_path)

        # Apply train/val/test split
        image_paths, mask_paths = self._apply_split(image_paths, mask_paths)

        return image_paths, mask_paths
    
    def _apply_split(
        self,
        image_paths: List[Path],
        mask_paths: List[Path]
    ) -> Tuple[List[Path], List[Path]]:
        """
        Split data into train/val/test sets.
        """
        # Check if split file exists
        split_file = self.data_dir / "split.json"
        
        if split_file.exists():
            with open(split_file, 'r') as f:
                splits = json.load(f)
            indices = splits[self.split]
        else:
            # Create splits
            n_samples = len(image_paths)
            indices = list(range(n_samples))
            
            # Split: 70% train, 15% val, 15% test
            train_idx, temp_idx = train_test_split(
                indices, test_size=0.3, random_state=42
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, random_state=42
            )
            
            splits = {
                'train': train_idx,
                'val': val_idx,
                'test': test_idx
            }
            
            # Save splits for reproducibility
            with open(split_file, 'w') as f:
                json.dump(splits, f, indent=2)
            
            indices = splits[self.split]
        
        # Filter paths based on indices
        image_paths = [image_paths[i] for i in indices]
        mask_paths = [mask_paths[i] for i in indices]
        
        return image_paths, mask_paths
    
    def _rgb_to_mask(self, rgb_mask: np.ndarray) -> np.ndarray:
        """
        Convert RGB color-coded mask to class indices.
        """
        height, width = rgb_mask.shape[:2]
        mask = np.zeros((height, width), dtype=np.int64)
        
        for color, class_id in self.color_map.items():
            # Create boolean mask for this color
            color_mask = np.all(rgb_mask == color, axis=-1)
            mask[color_mask] = class_id
        
        return mask
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Returns:
            tuple: (image, label) where:
                - image: Tensor of shape (3, H, W)
                - label: Tensor of shape (H, W) with class indices or point labels
        """
        # Check cache
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]
        
        # Load image and mask
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Read image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask
        mask = cv2.imread(str(mask_path))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # Convert RGB mask to class indices
        mask = self._rgb_to_mask(mask)
        
        # Resize if needed
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            mask = cv2.resize(
                mask,
                (self.image_size[1], self.image_size[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Sample point labels if needed
        if self.use_point_labels and self.point_sampler is not None:
            mask = self.point_sampler.sample_points(mask, self.num_classes)
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Default: normalize and convert to tensor
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = torch.from_numpy(mask).long()
        
        # Cache if enabled
        if self.cache is not None:
            self.cache[idx] = (image, mask)
        
        return image, mask
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalanced data.
        
        Returns:
            torch.Tensor: Class weights of shape (num_classes,)
        """
        class_counts = np.zeros(self.num_classes)
        
        print("Computing class weights from dataset...")
        for idx in range(len(self)):
            _, mask = self[idx]
            
            # Count pixels for each class
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            
            for class_id in range(self.num_classes):
                class_counts[class_id] += np.sum(mask == class_id)
        
        # Compute inverse frequency weights
        total_pixels = class_counts.sum()
        class_weights = total_pixels / (self.num_classes * class_counts + 1e-6)
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * self.num_classes
        
        return torch.from_numpy(class_weights).float()


def get_transforms(split: str = 'train', image_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """
    Get data augmentation transforms for different splits.
    
    Args:
        split: Dataset split ('train', 'val', 'test')
        image_size: Target image size
        
    Returns:
        A.Compose: Albumentations transform pipeline
    """
    if split == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    image_size: Tuple[int, int] = (512, 512),
    num_workers: int = 4,
    point_sampler: Optional[PointSampler] = None,
    use_augmentation: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_transform = get_transforms('train', image_size) if use_augmentation else get_transforms('val', image_size)
    val_transform = get_transforms('val', image_size)
    
    train_dataset = DeepGlobeDataset(
        data_dir=data_dir,
        split='train',
        image_size=image_size,
        point_sampler=point_sampler,
        transform=train_transform,
        use_point_labels=True,
    )
    
    val_dataset = DeepGlobeDataset(
        data_dir=data_dir,
        split='val',
        image_size=image_size,
        point_sampler=point_sampler,
        transform=val_transform,
        use_point_labels=True,
    )
    
    test_dataset = DeepGlobeDataset(
        data_dir=data_dir,
        split='test',
        image_size=image_size,
        point_sampler=None,  # Use dense masks for testing
        transform=val_transform,
        use_point_labels=False,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loader
    from point_sampling import PointSampler
    
    print("Testing DeepGlobe DataLoader...")
    
    # Create point sampler
    sampler = PointSampler(
        num_points_per_class=100,
        sampling_strategy='random',
        seed=42
    )
    
    # Test dataset
    try:
        dataset = DeepGlobeDataset(
            data_dir="data/raw",
            split='train',
            point_sampler=sampler,
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of classes: {dataset.num_classes}")
        print(f"Class names: {dataset.class_names}")
        
        # Test loading a sample
        image, mask = dataset[0]
        print(f"\nSample shape:")
        print(f"  Image: {image.shape}")
        print(f"  Mask: {mask.shape}")
        print(f"  Labeled points: {torch.sum(mask != -1).item()}")
        
        print("\nDataLoader test passed!")
    except Exception as e:
        print(f"Note: Full test requires dataset to be downloaded")
        print(f"Error: {e}")
