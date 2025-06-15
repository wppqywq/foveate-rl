"""
Foveal Vision Library
Implementation of biologically-inspired foveal vision for computer vision tasks.

This package implements:
- Foveal sampling transforms that mimic human retinal characteristics
- Multi-glimpse dataset builders for training and evaluation
- Efficiency metrics for computational and information processing evaluation
- Baseline models with fixed attention mechanisms
- Uncertainty-driven attention models (Phase 2)

Authors: Staircase Foveate Vision Project Team
Version: 2.0
"""

__version__ = "2.0.0"
__author__ = "Foveal Vision Project Team"

# Import core modules
from . import transforms
from . import dataset_builder
from . import metrics
from . import baseline_model

# Import key classes for easy access
from .transforms import FovealTransform, GlimpseNetwork
from .dataset_builder import MultiGlimpseDataset, create_baseline_dataloaders
from .metrics import EfficiencyMetrics, ModelComparator
from .baseline_model import FixedAttentionModel, create_baseline_model

__all__ = [
    # Modules
    'transforms',
    'dataset_builder', 
    'metrics',
    'baseline_model',
    
    # Key classes
    'FovealTransform',
    'GlimpseNetwork',
    'MultiGlimpseDataset',
    'create_baseline_dataloaders',
    'EfficiencyMetrics',
    'ModelComparator',
    'FixedAttentionModel',
    'create_baseline_model',
]

# Version info
def get_version():
    """Get package version."""
    return __version__

def get_info():
    """Get package information."""
    return {
        'name': 'fovea_lib',
        'version': __version__,
        'author': __author__,
        'description': 'Biologically-inspired foveal vision for computer vision'
    }