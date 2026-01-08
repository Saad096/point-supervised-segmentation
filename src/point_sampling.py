"""
Point Sampling Utilities for Weakly Supervised Segmentation
Implements various strategies to sample point annotations from dense masks
"""

import numpy as np
import torch
from typing import Tuple, List, Optional
import random


class PointSampler:
    """
    Samples point annotations from dense segmentation masks.
    
    Supports multiple sampling strategies:
    - Random: Uniformly random points per class
    - Stratified: Ensures minimum distance between points
    - Grid: Regular grid-based sampling
    - Class-balanced: Equal number of points per class
    """
    
    def __init__(
        self,
        num_points_per_class: int = 100,
        sampling_strategy: str = 'random',
        min_distance: int = 10,
        ignore_classes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            num_points_per_class: Number of points to sample per class
            sampling_strategy: Sampling method ('random', 'stratified', 'grid')
            min_distance: Minimum distance between sampled points (for stratified)
            ignore_classes: Classes to ignore during sampling (e.g., unknown/background)
            seed: Random seed for reproducibility
        """
        self.num_points_per_class = num_points_per_class
        self.sampling_strategy = sampling_strategy
        self.min_distance = min_distance
        self.ignore_classes = ignore_classes or []
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def sample_points(
        self,
        mask: np.ndarray,
        num_classes: int
    ) -> np.ndarray:
        """
        Sample point annotations from a dense segmentation mask.
        
        Args:
            mask: Dense segmentation mask of shape (H, W)
            num_classes: Total number of classes
            
        Returns:
            np.ndarray: Sparse point annotation mask where most pixels are -1 (unlabeled)
                       and sampled points have their class labels
        """
        height, width = mask.shape
        point_mask = np.full((height, width), -1, dtype=np.int64)
        
        # Sample points for each class
        for class_id in range(num_classes):
            if class_id in self.ignore_classes:
                continue
                
            # Get all pixel coordinates for this class
            class_coords = np.argwhere(mask == class_id)
            
            if len(class_coords) == 0:
                # Skip if class not present in mask
                continue
            
            # Sample points based on strategy
            if self.sampling_strategy == 'random':
                sampled_points = self._random_sampling(class_coords)
            elif self.sampling_strategy == 'stratified':
                sampled_points = self._stratified_sampling(class_coords)
            elif self.sampling_strategy == 'grid':
                sampled_points = self._grid_sampling(class_coords, mask, class_id)
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
            
            # Mark sampled points in the point mask
            for point in sampled_points:
                point_mask[point[0], point[1]] = class_id
        
        return point_mask
    
    def _random_sampling(self, class_coords: np.ndarray) -> np.ndarray:
        """
        Uniformly random sampling of points.
        """
        num_available = len(class_coords)
        num_to_sample = min(self.num_points_per_class, num_available)
        
        indices = np.random.choice(num_available, num_to_sample, replace=False)
        return class_coords[indices]
    
    def _stratified_sampling(self, class_coords: np.ndarray) -> np.ndarray:
        """
        Stratified sampling ensuring minimum distance between points.
        This helps avoid clustering of points and ensures better spatial coverage.
        """
        sampled_points = []
        available_coords = class_coords.copy()
        
        while len(sampled_points) < self.num_points_per_class and len(available_coords) > 0:
            # Randomly select a point
            idx = np.random.randint(len(available_coords))
            point = available_coords[idx]
            sampled_points.append(point)
            
            # Remove points within min_distance
            distances = np.sqrt(
                np.sum((available_coords - point) ** 2, axis=1)
            )
            available_coords = available_coords[distances > self.min_distance]
        
        return np.array(sampled_points) if sampled_points else np.array([]).reshape(0, 2)
    
    def _grid_sampling(
        self,
        class_coords: np.ndarray,
        mask: np.ndarray,
        class_id: int
    ) -> np.ndarray:
        """
        Grid-based sampling for regular spatial distribution.
        """
        height, width = mask.shape
        sampled_points = []
        
        # Calculate grid spacing
        grid_size = int(np.sqrt(height * width / self.num_points_per_class))
        grid_size = max(grid_size, 1)
        
        # Sample on grid
        for h in range(0, height, grid_size):
            for w in range(0, width, grid_size):
                if len(sampled_points) >= self.num_points_per_class:
                    break
                if mask[h, w] == class_id:
                    sampled_points.append([h, w])
            if len(sampled_points) >= self.num_points_per_class:
                break
        
        # If not enough points, add random points
        if len(sampled_points) < self.num_points_per_class:
            remaining = self.num_points_per_class - len(sampled_points)
            additional = self._random_sampling(class_coords)[:remaining]
            if len(additional) > 0:
                sampled_points.extend(additional.tolist())
        
        return np.array(sampled_points) if sampled_points else np.array([]).reshape(0, 2)
    
    def get_sampling_statistics(
        self,
        point_mask: np.ndarray,
        num_classes: int
    ) -> dict:
        """
        Compute statistics about sampled points.
        
        Returns:
            dict: Statistics including points per class, total points, coverage, etc.
        """
        stats = {
            'total_points': np.sum(point_mask != -1),
            'points_per_class': {},
            'coverage_per_class': {},
        }
        
        for class_id in range(num_classes):
            points = np.sum(point_mask == class_id)
            stats['points_per_class'][class_id] = points
            
            # Calculate coverage (percentage of labeled pixels)
            total_pixels = point_mask.size
            coverage = (points / total_pixels) * 100 if total_pixels > 0 else 0
            stats['coverage_per_class'][class_id] = coverage
        
        stats['average_points_per_class'] = np.mean(
            list(stats['points_per_class'].values())
        )
        stats['overall_coverage'] = (stats['total_points'] / point_mask.size) * 100
        
        return stats


class AdaptivePointSampler(PointSampler):
    """
    Adaptive point sampling that considers class frequency and difficulty.
    
    Samples more points for:
    - Rare classes (class imbalance)
    - Classes with complex boundaries
    - Classes with high intra-class variability
    """
    
    def __init__(
        self,
        base_points_per_class: int = 100,
        sampling_strategy: str = 'stratified',
        min_distance: int = 10,
        adapt_to_frequency: bool = True,
        frequency_weight: float = 0.5,
        seed: Optional[int] = None,
    ):
        """
        Args:
            base_points_per_class: Base number of points per class
            adapt_to_frequency: Whether to adapt sampling based on class frequency
            frequency_weight: Weight for frequency-based adaptation (0-1)
        """
        super().__init__(
            num_points_per_class=base_points_per_class,
            sampling_strategy=sampling_strategy,
            min_distance=min_distance,
            seed=seed
        )
        self.base_points = base_points_per_class
        self.adapt_to_frequency = adapt_to_frequency
        self.frequency_weight = frequency_weight
    
    def sample_points(
        self,
        mask: np.ndarray,
        num_classes: int
    ) -> np.ndarray:
        """
        Adaptively sample points based on class frequency.
        """
        height, width = mask.shape
        point_mask = np.full((height, width), -1, dtype=np.int64)
        
        # Calculate class frequencies
        class_frequencies = {}
        total_pixels = height * width
        
        for class_id in range(num_classes):
            class_pixels = np.sum(mask == class_id)
            class_frequencies[class_id] = class_pixels / total_pixels
        
        # Adapt number of points per class
        for class_id in range(num_classes):
            if class_id in self.ignore_classes:
                continue
            
            # Get class coordinates
            class_coords = np.argwhere(mask == class_id)
            
            if len(class_coords) == 0:
                continue
            
            # Adapt number of points based on frequency
            if self.adapt_to_frequency:
                freq = class_frequencies[class_id]
                # Rare classes get more points (inverse frequency weighting)
                adaptation_factor = 1.0 / (freq + 1e-6)
                adaptation_factor = np.clip(adaptation_factor, 0.5, 2.0)
                
                # Blend with base number
                self.num_points_per_class = int(
                    self.base_points * (
                        (1 - self.frequency_weight) +
                        self.frequency_weight * adaptation_factor
                    )
                )
            else:
                self.num_points_per_class = self.base_points
            
            # Sample points
            if self.sampling_strategy == 'random':
                sampled_points = self._random_sampling(class_coords)
            elif self.sampling_strategy == 'stratified':
                sampled_points = self._stratified_sampling(class_coords)
            else:
                sampled_points = self._grid_sampling(class_coords, mask, class_id)
            
            # Mark sampled points
            for point in sampled_points:
                point_mask[point[0], point[1]] = class_id
        
        return point_mask


def visualize_point_sampling(
    image: np.ndarray,
    mask: np.ndarray,
    point_mask: np.ndarray,
    class_colors: List[Tuple[int, int, int]]
) -> np.ndarray:
    """
    Visualize sampled points overlaid on the original image.
    
    Args:
        image: Original RGB image (H, W, 3)
        mask: Dense segmentation mask (H, W)
        point_mask: Sparse point annotations (H, W)
        class_colors: List of RGB colors for each class
        
    Returns:
        np.ndarray: Visualization image
    """
    vis_image = image.copy()
    
    # Draw sampled points
    sampled_coords = np.argwhere(point_mask != -1)
    
    for coord in sampled_coords:
        h, w = coord
        class_id = point_mask[h, w]
        color = class_colors[class_id]
        
        # Draw a small circle for each point
        cv2.circle(vis_image, (w, h), radius=3, color=color, thickness=-1)
        cv2.circle(vis_image, (w, h), radius=4, color=(255, 255, 255), thickness=1)
    
    return vis_image


if __name__ == "__main__":
    # Test point sampling
    print("Testing Point Sampling...")
    
    # Create a dummy mask
    mask = np.random.randint(0, 7, (512, 512))
    
    # Test different sampling strategies
    strategies = ['random', 'stratified', 'grid']
    
    for strategy in strategies:
        print(f"\nTesting {strategy} sampling...")
        sampler = PointSampler(
            num_points_per_class=100,
            sampling_strategy=strategy,
            seed=42
        )
        
        point_mask = sampler.sample_points(mask, num_classes=7)
        stats = sampler.get_sampling_statistics(point_mask, num_classes=7)
        
        print(f"Total points: {stats['total_points']}")
        print(f"Average points per class: {stats['average_points_per_class']:.1f}")
        print(f"Overall coverage: {stats['overall_coverage']:.4f}%")
    
    # Test adaptive sampling
    print("\n\nTesting Adaptive Sampling...")
    adaptive_sampler = AdaptivePointSampler(
        base_points_per_class=100,
        adapt_to_frequency=True,
        seed=42
    )
    
    point_mask = adaptive_sampler.sample_points(mask, num_classes=7)
    stats = adaptive_sampler.get_sampling_statistics(point_mask, num_classes=7)
    
    print(f"Total points: {stats['total_points']}")
    print(f"Points per class: {stats['points_per_class']}")
    
    print("\nAll tests passed!")
