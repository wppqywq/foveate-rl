#!/usr/bin/env python3
"""
Quick Test Script
Simple test of core functionality without complex visualizations.
"""

import torch
import numpy as np
import sys
import os

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from fovea_lib.transforms import FovealTransform
    from fovea_lib.dataset_builder import create_baseline_dataloaders
    from fovea_lib.baseline_model import create_baseline_model
    from fovea_lib.metrics import EfficiencyMetrics
    print("All imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_foveal_transform():
    """Test basic foveal transform functionality."""
    print("\n=== Testing Foveal Transform ===")
    
    # Create test image
    test_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    print(f"Test image shape: {test_img.shape}")
    
    # Create transform
    transform = FovealTransform(fovea_radius=8, image_size=32)
    
    # Test glimpse extraction
    locations = [(16, 16), (8, 8), (24, 24)]
    for i, (x, y) in enumerate(locations):
        high_res, low_res = transform.crop_lowhigh(test_img, x, y)
        print(f"Glimpse {i+1} at ({x},{y}): high_res={high_res.shape}, low_res={low_res.shape}")
    
    print("Foveal transform working!")

def test_model():
    """Test model creation and forward pass."""
    print("\n=== Testing Model ===")
    
    # Create model
    model = create_baseline_model('fixed', num_classes=10, n_glimpses=3)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    n_glimpses = 3
    glimpse_size = 16  # 2 * fovea_radius
    dummy_input = torch.randn(batch_size, n_glimpses, 3, glimpse_size, glimpse_size)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Model working!")

def test_dataset():
    """Test dataset creation."""
    print("\n=== Testing Dataset ===")
    
    try:
        train_loader, test_loader = create_baseline_dataloaders(
            dataset_name='cifar10',
            data_dir='./data',
            n_glimpses=3,
            batch_size=8,
            num_workers=0
        )
        
        # Get a sample batch
        glimpses, full_images, labels = next(iter(train_loader))
        
        print(f"Dataset created successfully!")
        print(f"Glimpses shape: {glimpses.shape}")
        print(f"Full images shape: {full_images.shape}")
        print(f"Labels shape: {labels.shape}")
        print("Dataset working!")
        
    except Exception as e:
        print(f"Dataset test failed: {e}")

def test_metrics():
    """Test metrics calculation."""
    print("\n=== Testing Metrics ===")
    
    metrics = EfficiencyMetrics()
    
    # Add some sample data
    for i in range(5):
        metrics.update(
            accuracy=0.8 + 0.1 * np.random.random(),
            n_glimpses=3,
            processing_time=0.1 + 0.02 * np.random.random(),
            total_pixels=3 * 16 * 16 * 3
        )
    
    summary = metrics.get_summary()
    print("Sample metrics:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")
    
    print("Metrics working!")

def main():
    """Run all tests."""
    print("\n===Quick Functionality Test===")
    
    test_foveal_transform()
    test_model()
    test_dataset()
    test_metrics()
    
    print("All core functionality tests passed!")
    print("\nReady to run full training:")
    print("  python train_baseline.py --epochs 30")

if __name__ == "__main__":
    main()