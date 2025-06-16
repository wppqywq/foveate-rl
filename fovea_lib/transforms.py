"""
Foveal Vision Transform Implementation
Multi-resolution visual sampling with eccentricity-dependent resolution.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
from PIL import Image


class FovealTransform:
    """
    Implements biologically-inspired foveal sampling.
    Resolution decreases with distance from fixation point.
    """
    
    def __init__(self, 
                 image_size: int = 32,
                 fovea_radius: int = 4,
                 n_rings: int = 3,
                 scale_factor: float = 2.0):
        """
        Args:
            image_size: Input image size
            fovea_radius: Radius of central high-resolution region
            n_rings: Number of concentric rings with decreasing resolution
            scale_factor: Resolution reduction factor per ring
        """
        self.image_size = image_size
        self.fovea_radius = fovea_radius
        self.n_rings = n_rings
        self.scale_factor = scale_factor
        
        # Precompute ring boundaries and resolutions
        self.ring_boundaries = []
        self.ring_resolutions = []
        
        for i in range(n_rings):
            boundary = fovea_radius * (scale_factor ** i)
            resolution = 1.0 / (scale_factor ** i)
            self.ring_boundaries.append(boundary)
            self.ring_resolutions.append(resolution)
    
    def get_resolution_at_distance(self, distance: float) -> float:
        """Get resolution factor based on distance from fixation."""
        for i, boundary in enumerate(self.ring_boundaries):
            if distance <= boundary:
                return self.ring_resolutions[i]
        return self.ring_resolutions[-1]
    
    def create_foveal_sample(self, 
                           image: np.ndarray, 
                           fixation_x: int, 
                           fixation_y: int,
                           output_size: int = 64) -> np.ndarray:
        """
        Create foveated sample with resolution decreasing by eccentricity.
        
        Args:
            image: Input image (H, W, C)
            fixation_x, fixation_y: Fixation point coordinates
            output_size: Size of output foveated image
            
        Returns:
            Foveated sample with spatially varying resolution
        """
        h, w = image.shape[:2]
        foveated = np.zeros((output_size, output_size, 3), dtype=np.uint8)
        
        # Create coordinate grids
        y_coords, x_coords = np.meshgrid(
            np.linspace(0, h-1, output_size),
            np.linspace(0, w-1, output_size),
            indexing='ij'
        )
        
        # Map output coordinates to input coordinates
        scale_x = w / output_size
        scale_y = h / output_size
        
        for i in range(output_size):
            for j in range(output_size):
                # Distance from fixation in original image coordinates
                img_x = j * scale_x
                img_y = i * scale_y
                distance = np.sqrt((img_x - fixation_x)**2 + (img_y - fixation_y)**2)
                
                # Get resolution factor for this distance
                resolution = self.get_resolution_at_distance(distance)
                
                # Sample with appropriate resolution
                if resolution >= 1.0:
                    # High resolution - direct sampling
                    x_idx = int(np.clip(img_x, 0, w-1))
                    y_idx = int(np.clip(img_y, 0, h-1))
                    foveated[i, j] = image[y_idx, x_idx]
                else:
                    # Low resolution - average over neighborhood
                    kernel_size = int(1.0 / resolution)
                    x_start = int(np.clip(img_x - kernel_size//2, 0, w-kernel_size))
                    x_end = x_start + kernel_size
                    y_start = int(np.clip(img_y - kernel_size//2, 0, h-kernel_size))
                    y_end = y_start + kernel_size
                    
                    patch = image[y_start:y_end, x_start:x_end]
                    foveated[i, j] = np.mean(patch, axis=(0, 1))
        
        return foveated
    
    def create_glimpse_tensor(self, 
                            image: np.ndarray,
                            fixation: Tuple[int, int],
                            glimpse_size: int = 64) -> torch.Tensor:
        """Create normalized glimpse tensor."""
        foveated = self.create_foveal_sample(
            image, fixation[0], fixation[1], glimpse_size
        )
        
        # Convert to tensor and normalize
        tensor = torch.from_numpy(foveated).permute(2, 0, 1).float() / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor


class GlimpseNetwork(torch.nn.Module):
    """Encodes glimpses into feature representations."""
    
    def __init__(self, glimpse_size: int = 64, feature_dim: int = 256):
        super().__init__()
        
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128 * 16, feature_dim),
            torch.nn.ReLU()
        )
        
    def forward(self, glimpse: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(glimpse)
        features = features.view(features.size(0), -1)
        return self.fc(features)


def visualize_foveal_pattern(image: np.ndarray, 
                           fixations: List[Tuple[int, int]],
                           transform: FovealTransform) -> np.ndarray:
    """Visualize foveal sampling pattern."""
    vis_img = image.copy()
    
    for i, (x, y) in enumerate(fixations):
        # Draw fixation point
        cv2.circle(vis_img, (x, y), 2, (0, 255, 0), -1)
        
        # Draw foveal rings
        for ring_idx, boundary in enumerate(transform.ring_boundaries):
            color = (255 - ring_idx * 60, 255 - ring_idx * 60, 0)
            cv2.circle(vis_img, (x, y), int(boundary), color, 1)
    
    return vis_img