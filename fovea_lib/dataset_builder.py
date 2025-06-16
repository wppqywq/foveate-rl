"""
Dataset builders for attention learning experiments.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple


class AttentionDataset(Dataset):
    """Dataset wrapper for RL attention training."""
    
    def __init__(self, base_dataset: Dataset, augment: bool = True):
        self.base_dataset = base_dataset
        
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.base_dataset[idx]
        image = self.transform(image)
        return image, label


def create_cifar10_loaders(data_dir: str = './data',
                          batch_size: int = 128,
                          num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 dataloaders for attention learning."""
    
    base_transform = transforms.Resize((32, 32))
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=base_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=base_transform
    )
    
    train_attention = AttentionDataset(train_dataset, augment=True)
    test_attention = AttentionDataset(test_dataset, augment=False)
    
    train_loader = DataLoader(
        train_attention, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_attention, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def create_mnist_loaders(data_dir: str = './data',
                        batch_size: int = 128,
                        num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Create MNIST dataloaders for attention learning."""
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Lambda(lambda x: x.convert('RGB')),  # Convert to RGB
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader