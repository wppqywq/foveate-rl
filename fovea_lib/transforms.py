"""
Foveal Vision Transform Implementation
Implements multi-resolution visual sampling that mimics human retinal sampling characteristics.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Tuple, Union, Optional
from PIL import Image


def simple_downsample(img: np.ndarray, factor: float) -> np.ndarray:
    """
    Simple downsampling using numpy array slicing.
    Fallback when both OpenCV and PIL fail.
    
    Args:
        img: Input image
        factor: Downsampling factor
        
    Returns:
        Downsampled image
    """
    h, w = img.shape[:2]
    new_h = int(h / factor)
    new_w = int(w / factor)
    
    # Simple decimation (take every nth pixel)
    step_h = max(1, h // new_h)
    step_w = max(1, w // new_w)
    
    if len(img.shape) == 3:
        downsampled = img[::step_h, ::step_w, :][:new_h, :new_w, :]
    else:
        downsampled = img[::step_h, ::step_w][:new_h, :new_w]
    
    return downsampled


class FovealTransform:
    """
    Implements foveal vision sampling strategy.
    High resolution in the center (fovea), decreasing resolution towards periphery.
    """
    
    def __init__(self, 
                 fovea_radius: int = 16,
                 scale_factor: float = 4.0,
                 image_size: int = 32):
        """
        Initialize foveal transform parameters.
        
        Args:
            fovea_radius: Radius of high-resolution foveal region
            scale_factor: Downsampling factor for peripheral region
            image_size: Size of input images (assumes square images)
        """
        self.fovea_radius = fovea_radius
        self.scale_factor = scale_factor
        self.image_size = image_size
        
    def crop_lowhigh(self, 
                     img: np.ndarray, 
                     x: int, 
                     y: int, 
                     r: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract foveal patch and create low-resolution background.
        
        Args:
            img: Input image as HxWxC numpy array (0-255 uint8)
            x, y: Center coordinates for foveal region
            r: Radius of foveal region (defaults to self.fovea_radius)
            
        Returns:
            high_res_patch: r x r high-resolution foveal region
            low_res_background: Downsampled background image
        """
        if r is None:
            r = self.fovea_radius
        
        # Ensure image is in correct format
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        
        # Ensure image is contiguous and in uint8 format
        img = np.ascontiguousarray(img, dtype=np.uint8)
            
        h, w = img.shape[:2]
        
        # Ensure coordinates are within bounds
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        
        # Extract high-resolution foveal patch
        x1, x2 = x - r, x + r
        y1, y2 = y - r, y + r
        high_res_patch = img[y1:y2, x1:x2].copy()
        
    def crop_lowhigh(self, 
                     img: np.ndarray, 
                     x: int, 
                     y: int, 
                     r: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract foveal patch and create low-resolution background.
        
        Args:
            img: Input image as HxWxC numpy array (0-255 uint8)
            x, y: Center coordinates for foveal region
            r: Radius of foveal region (defaults to self.fovea_radius)
            
        Returns:
            high_res_patch: r x r high-resolution foveal region
            low_res_background: Downsampled background image
        """
        if r is None:
            r = self.fovea_radius
        
        # Ensure image is in correct format
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        
        # Ensure image is contiguous and in uint8 format
        img = np.ascontiguousarray(img, dtype=np.uint8)
            
        h, w = img.shape[:2]
        
        # Ensure coordinates are within bounds
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        
        # Extract high-resolution foveal patch
        x1, x2 = x - r, x + r
        y1, y2 = y - r, y + r
        high_res_patch = img[y1:y2, x1:x2].copy()
        
        # Create low-resolution background using PIL (OpenCV 4.11.0 has compatibility issues)
        low_res_h = int(h / self.scale_factor)
        low_res_w = int(w / self.scale_factor)
        
        try:
            # Use PIL as primary method (more reliable on macOS)
            from PIL import Image
            
            # Convert to PIL Image
            if len(img.shape) == 3:
                pil_img = Image.fromarray(img, 'RGB')
            else:
                pil_img = Image.fromarray(img, 'L')
            
            # Resize using PIL with high-quality resampling
            pil_resized = pil_img.resize((low_res_w, low_res_h), Image.LANCZOS)
            low_res_background = np.array(pil_resized)
            
        except Exception as e_pil:
            # Fallback: simple numpy decimation
            print(f"PIL resize failed ({e_pil}), using simple decimation...")
            low_res_background = simple_downsample(img, self.scale_factor)
        
        return high_res_patch, low_res_background
        
        return high_res_patch, low_res_background
    
    def create_multi_glimpse_batch(self, 
                                   img: np.ndarray, 
                                   locations: list) -> torch.Tensor:
        """
        Create a batch of glimpses from multiple fixation locations.
        
        Args:
            img: Input image as HxWxC numpy array
            locations: List of (x, y) tuples for fixation points
            
        Returns:
            glimpse_batch: Tensor of shape (n_glimpses, C, 2*r, 2*r)
        """
        glimpses = []
        for x, y in locations:
            high_res_patch, _ = self.crop_lowhigh(img, x, y)
            # Convert to tensor and normalize
            if len(high_res_patch.shape) == 3:
                glimpse = torch.from_numpy(high_res_patch).permute(2, 0, 1).float() / 255.0
            else:
                glimpse = torch.from_numpy(high_res_patch).unsqueeze(0).float() / 255.0
            glimpses.append(glimpse)
        
        return torch.stack(glimpses)
    
    def generate_fixed_locations(self, n_glimpses: int = 3) -> list:
        """
        Generate fixed glimpse locations for baseline model.
        
        Args:
            n_glimpses: Number of glimpse locations to generate
            
        Returns:
            List of (x, y) coordinates
        """
        locations = []
        center = self.image_size // 2
        
        if n_glimpses == 1:
            locations = [(center, center)]
        elif n_glimpses == 3:
            # Center + two offset locations
            offset = self.image_size // 4
            locations = [
                (center, center),
                (center - offset, center - offset),
                (center + offset, center + offset)
            ]
        elif n_glimpses == 5:
            # Center + four corners pattern
            offset = self.image_size // 3
            locations = [
                (center, center),
                (center - offset, center - offset),
                (center + offset, center - offset),
                (center - offset, center + offset),
                (center + offset, center + offset)
            ]
        else:
            # Random grid sampling for other cases
            grid_size = int(np.sqrt(n_glimpses))
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(locations) >= n_glimpses:
                        break
                    x = (i + 1) * self.image_size // (grid_size + 1)
                    y = (j + 1) * self.image_size // (grid_size + 1)
                    locations.append((x, y))
        
        return locations[:n_glimpses]


class GlimpseNetwork(torch.nn.Module):
    """
    Neural network module for processing glimpses.
    Encodes high-resolution patches into feature representations.
    """
    
    def __init__(self, 
                 glimpse_size: int = 32,
                 hidden_size: int = 256,
                 num_channels: int = 3):
        """
        Initialize glimpse network.
        
        Args:
            glimpse_size: Size of input glimpse patches
            hidden_size: Size of hidden representations
            num_channels: Number of input channels (3 for RGB)
        """
        super().__init__()
        
        self.glimpse_size = glimpse_size
        self.hidden_size = hidden_size
        
        # Convolutional feature extractor
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Fully connected layers
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(128 * 4 * 4, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, glimpses: torch.Tensor) -> torch.Tensor:
        """
        Process glimpses through the network.
        
        Args:
            glimpses: Tensor of shape (batch_size, n_glimpses, C, H, W)
            
        Returns:
            features: Tensor of shape (batch_size, n_glimpses, hidden_size)
        """
        batch_size, n_glimpses = glimpses.shape[:2]
        
        # Reshape for batch processing
        x = glimpses.view(-1, *glimpses.shape[2:])
        
        # Extract features
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        features = self.fc_layers(x)
        
        # Reshape back to original batch structure
        features = features.view(batch_size, n_glimpses, -1)
        
        return features


def visualize_foveal_sampling(img: np.ndarray, 
                            locations: list, 
                            fovea_radius: int = 16) -> np.ndarray:
    """
    Visualize foveal sampling locations on an image.
    Simple and reliable visualization method.
    
    Args:
        img: Input image
        locations: List of (x, y) fixation points
        fovea_radius: Radius of foveal regions
        
    Returns:
        Annotated image with fixation points marked
    """
    vis_img = img.copy()
    
    # Simple and reliable method: draw colored crosses at fixation points
    for i, (x, y) in enumerate(locations):
        # Draw a cross pattern to mark fixation points
        cross_size = max(3, fovea_radius // 4)
        
        # Horizontal line
        x1, x2 = max(0, x-cross_size), min(img.shape[1], x+cross_size+1)
        if 0 <= y < img.shape[0]:
            vis_img[y, x1:x2] = [0, 255, 0]  # Green horizontal line
        
        # Vertical line  
        y1, y2 = max(0, y-cross_size), min(img.shape[0], y+cross_size+1)
        if 0 <= x < img.shape[1]:
            vis_img[y1:y2, x] = [0, 255, 0]  # Green vertical line
        
        # Add a small colored square for the fixation number
        square_size = 2
        for dy in range(-square_size, square_size+1):
            for dx in range(-square_size, square_size+1):
                px, py = x + dx + 5, y + dy - 5
                if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                    # Use different colors for different fixations
                    if i == 0:
                        vis_img[py, px] = [255, 0, 0]  # Red for fixation 1
                    elif i == 1:
                        vis_img[py, px] = [0, 0, 255]  # Blue for fixation 2  
                    else:
                        vis_img[py, px] = [255, 255, 0]  # Yellow for fixation 3+
    
    return vis_img