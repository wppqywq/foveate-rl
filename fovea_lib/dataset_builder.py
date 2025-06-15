"""
Multi-Glimpse Dataset Builder
Implements multi-glimpse representation learning for foveal vision experiments.
Creates datasets with multiple fixed-position glimpses per image.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List, Optional
from .transforms import FovealTransform


class MultiGlimpseDataset(Dataset):
    """
    Dataset wrapper that generates multiple glimpses per image.
    Each sample contains multiple fixed-position glimpses for training baseline models.
    """
    
    def __init__(self, 
                 base_dataset: Dataset,
                 n_glimpses: int = 3,
                 glimpse_size: int = 32,
                 fovea_radius: int = 8,  # Changed from 16 to 8 for actual glimpses
                 image_size: int = 32,
                 augment: bool = True):
        """
        Initialize multi-glimpse dataset.
        
        Args:
            base_dataset: Underlying dataset (e.g., CIFAR-10)
            n_glimpses: Number of glimpses per image
            glimpse_size: Size of each glimpse
            fovea_radius: Radius of foveal region
            image_size: Size of input images
            augment: Whether to apply data augmentation
        """
        self.base_dataset = base_dataset
        self.n_glimpses = n_glimpses
        self.foveal_transform = FovealTransform(
            fovea_radius=fovea_radius,
            image_size=image_size
        )
        
        # Generate fixed glimpse locations
        self.glimpse_locations = self.foveal_transform.generate_fixed_locations(n_glimpses)
        
        # Data augmentation transforms
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self.augment_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get item with multiple glimpses.
        
        Args:
            idx: Index of the sample
            
        Returns:
            glimpses: Tensor of shape (n_glimpses, C, H, W)
            full_image: Full resolution image tensor
            label: Class label
        """
        image, label = self.base_dataset[idx]
        
        # Convert PIL image to numpy if needed
        if hasattr(image, 'convert'):
            image_np = np.array(image.convert('RGB'))
        else:
            image_np = image
        
        # Generate glimpses
        glimpses = self.foveal_transform.create_multi_glimpse_batch(
            image_np, self.glimpse_locations
        )
        
        # Apply augmentation to full image
        full_image = self.augment_transform(image)
        
        return glimpses, full_image, label


class CIFAR10MultiGlimpse:
    """
    CIFAR-10 dataset builder with multi-glimpse support.
    """
    
    @staticmethod
    def get_datasets(data_dir: str = './data',
                    n_glimpses: int = 3,
                    batch_size: int = 128,
                    num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
        """
        Create CIFAR-10 train and test dataloaders with multi-glimpse support.
        
        Args:
            data_dir: Directory to store dataset
            n_glimpses: Number of glimpses per image
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            
        Returns:
            train_loader: Training data loader
            test_loader: Test data loader
        """
        
        # Base transforms for CIFAR-10
        base_transform = transforms.Compose([
            transforms.Resize((32, 32)),
        ])
        
        # Load base datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=base_transform
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=base_transform
        )
        
        # Wrap with multi-glimpse functionality
        train_multi_glimpse = MultiGlimpseDataset(
            train_dataset,
            n_glimpses=n_glimpses,
            augment=True
        )
        
        test_multi_glimpse = MultiGlimpseDataset(
            test_dataset,
            n_glimpses=n_glimpses,
            augment=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_multi_glimpse,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_multi_glimpse,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader


class AdaptiveGlimpseDataset(Dataset):
    """
    Dataset for uncertainty-driven attention experiments.
    Supports dynamic glimpse generation during training.
    """
    
    def __init__(self,
                 base_dataset: Dataset,
                 max_glimpses: int = 5,
                 image_size: int = 32,
                 fovea_radius: int = 16):
        """
        Initialize adaptive glimpse dataset.
        
        Args:
            base_dataset: Underlying dataset
            max_glimpses: Maximum number of glimpses allowed
            image_size: Size of input images
            fovea_radius: Radius of foveal region
        """
        self.base_dataset = base_dataset
        self.max_glimpses = max_glimpses
        self.foveal_transform = FovealTransform(
            fovea_radius=fovea_radius,
            image_size=image_size
        )
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Get raw image and label for dynamic glimpse generation.
        
        Args:
            idx: Index of the sample
            
        Returns:
            image: Raw image as numpy array
            label: Class label
        """
        image, label = self.base_dataset[idx]
        
        # Convert to numpy array
        if hasattr(image, 'convert'):
            image_np = np.array(image.convert('RGB'))
        else:
            image_np = image
        
        return image_np, label
    
    def extract_glimpse_at_location(self, 
                                   image: np.ndarray, 
                                   x: int, 
                                   y: int) -> torch.Tensor:
        """
        Extract a single glimpse at specified location.
        
        Args:
            image: Input image as numpy array
            x, y: Glimpse center coordinates
            
        Returns:
            glimpse: Normalized glimpse tensor
        """
        high_res_patch, _ = self.foveal_transform.crop_lowhigh(image, x, y)
        
        # Convert to tensor and normalize
        if len(high_res_patch.shape) == 3:
            glimpse = torch.from_numpy(high_res_patch).permute(2, 0, 1).float() / 255.0
        else:
            glimpse = torch.from_numpy(high_res_patch).unsqueeze(0).float() / 255.0
        
        # Apply normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        glimpse = (glimpse - mean) / std
        
        return glimpse


def create_baseline_dataloaders(dataset_name: str = 'cifar10',
                              data_dir: str = './data',
                              n_glimpses: int = 3,
                              batch_size: int = 128,
                              num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create baseline dataloaders for different datasets.
    
    Args:
        dataset_name: Name of the dataset ('cifar10', 'mnist', etc.)
        data_dir: Directory to store dataset
        n_glimpses: Number of glimpses per image
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        train_loader: Training data loader
        test_loader: Test data loader
    """
    
    if dataset_name.lower() == 'cifar10':
        return CIFAR10MultiGlimpse.get_datasets(
            data_dir=data_dir,
            n_glimpses=n_glimpses,
            batch_size=batch_size,
            num_workers=num_workers
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet")


def collate_glimpses(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for multi-glimpse batches.
    
    Args:
        batch: List of (glimpses, full_image, label) tuples
        
    Returns:
        glimpses_batch: Batched glimpses tensor
        images_batch: Batched full images tensor
        labels_batch: Batched labels tensor
    """
    glimpses, images, labels = zip(*batch)
    
    glimpses_batch = torch.stack(glimpses)
    images_batch = torch.stack(images)
    labels_batch = torch.tensor(labels)
    
    return glimpses_batch, images_batch, labels_batch