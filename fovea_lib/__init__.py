"""
Foveal Vision Library
Implementation of RL-based attention learning for foveal pattern emergence.
"""

__version__ = "2.0.0"

from .transforms import FovealTransform, GlimpseNetwork
from .rl_attention import RecurrentAttentionModel, REINFORCETrainer, LocationNetwork
from .dataset_builder import create_cifar10_loaders, create_mnist_loaders, AttentionDataset
from .metrics import EmergenceAnalyzer, TrainingMonitor, compute_attention_statistics

__all__ = [
    'FovealTransform',
    'GlimpseNetwork', 
    'RecurrentAttentionModel',
    'REINFORCETrainer',
    'LocationNetwork',
    'create_cifar10_loaders',
    'create_mnist_loaders',
    'AttentionDataset',
    'EmergenceAnalyzer',
    'TrainingMonitor',
    'compute_attention_statistics'
]