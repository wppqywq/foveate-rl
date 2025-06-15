"""
Phase 1 Setup and Demo Script
Sets up the environment and runs a quick demonstration of the foveal vision baseline.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from pathlib import Path
import cv2

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

try:
    from fovea_lib.transforms import FovealTransform, visualize_foveal_sampling
    from fovea_lib.dataset_builder import create_baseline_dataloaders
    from fovea_lib.baseline_model import create_baseline_model
    from fovea_lib.metrics import EfficiencyMetrics
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the foveate directory and all files are in fovea_lib/")
    sys.exit(1)


def check_environment():
    """Check if the environment is properly set up."""
    print("=== Environment Check ===")
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
    except ImportError:
        print("PyTorch not installed!")
        return False
    
    # Other dependencies
    dependencies = ['torchvision', 'numpy', 'matplotlib', 'opencv-python', 'tqdm', 'PIL']
    missing = []
    
    for dep in dependencies:
        try:
            if dep == 'opencv-python':
                import cv2
            elif dep == 'PIL':
                from PIL import Image
            else:
                __import__(dep)
            print(f"{dep}: OK")
        except ImportError:
            print(f"{dep}: Missing")
            missing.append(dep)
    
    if missing:
        print(f"\nMissing dependencies: {missing}")
        print("Please install them with: pip install " + " ".join(missing))
        return False
    
    print("Environment check passed!")
    return True


def demo_foveal_sampling():
    """Demonstrate foveal sampling on a sample image."""
    print("\n=== Foveal Sampling Demo ===")
    
    # Create a sample image or use CIFAR-10
    try:
        from torchvision.datasets import CIFAR10
        from torchvision import transforms
        
        # Download a sample from CIFAR-10
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        # Get a sample image
        sample_idx = 42  # Fixed sample for reproducibility
        image_tensor, label = dataset[sample_idx]
        
        # Convert to numpy - ensure proper format for OpenCV
        image_np = image_tensor.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
        image_np = np.ascontiguousarray(image_np)
        
        print(f"Sample image shape: {image_np.shape}")
        print(f"Sample image dtype: {image_np.dtype}")
        print(f"Sample image value range: [{image_np.min()}, {image_np.max()}]")
        print(f"Sample label: {label} ({dataset.classes[label]})")
        
    except Exception as e:
        print(f"Could not load CIFAR-10: {e}")
        # Create a synthetic image
        image_np = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        print("Using synthetic image for demo")
    
    # Initialize foveal transform
    foveal_transform = FovealTransform(fovea_radius=8, image_size=32)
    
    # Generate glimpse locations
    locations = foveal_transform.generate_fixed_locations(n_glimpses=3)
    print(f"Glimpse locations: {locations}")
    
    # Test one glimpse first to debug
    print(f"Testing single glimpse extraction...")
    try:
        x, y = locations[0]
        print(f"Extracting glimpse at ({x}, {y})")
        high_res_patch, low_res_bg = foveal_transform.crop_lowhigh(image_np, x, y)
        print(f"Single glimpse extraction successful!")
        print(f"High-res patch shape: {high_res_patch.shape}")
        print(f"Low-res background shape: {low_res_bg.shape}")
    except Exception as e:
        print(f"Single glimpse extraction failed: {e}")
        print("This indicates an issue with the foveal transform. Skipping visualization...")
        return
    
    # Extract all glimpses
    glimpses = []
    for i, (x, y) in enumerate(locations):
        print(f"Extracting glimpse {i+1} at ({x}, {y})")
        high_res_patch, low_res_bg = foveal_transform.crop_lowhigh(image_np, x, y)
        glimpses.append(high_res_patch)
    
    print(f"All glimpses extracted successfully!")
    
    # Visualize
    try:
        # Set matplotlib to non-interactive mode to avoid display issues
        plt.ioff()
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # Original image with fixation points
        vis_img = visualize_foveal_sampling(image_np, locations, fovea_radius=8)
        axes[0, 0].imshow(vis_img)
        axes[0, 0].set_title('Original with Fixations')
        axes[0, 0].axis('off')
        
        # Show low-res background
        _, low_res_bg = foveal_transform.crop_lowhigh(image_np, 16, 16)
        axes[0, 1].imshow(low_res_bg)
        axes[0, 1].set_title('Low-res Background')
        axes[0, 1].axis('off')
        
        # Show individual glimpses
        for i, glimpse in enumerate(glimpses):
            if i == 0:
                row, col = 0, 2
            elif i == 1:
                row, col = 1, 0
            else:  # i == 2
                row, col = 1, 1
            
            axes[row, col].imshow(glimpse)
            axes[row, col].set_title(f'Glimpse {i+1}')
            axes[row, col].axis('off')
        
        # Hide unused subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save instead of show to avoid display issues
        save_path = './foveal_sampling_demo.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()  # Close to free memory
        
        print("Foveal sampling demo completed!")
        print(f"Demo visualization saved as '{save_path}'")
        
        # Verify file was created
        if os.path.exists(save_path):
            print(f"File size: {os.path.getsize(save_path)} bytes")
        else:
            print("Warning: Visualization file was not created")
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("The core functionality works, but visualization has issues.")
        print("You can run 'python experiments/quick_test.py' for a simple functionality test.")


def demo_model_creation():
    """Demonstrate model creation and forward pass."""
    print("\n=== Model Creation Demo ===")
    
    # Create baseline model
    model = create_baseline_model(
        model_type='fixed',
        num_classes=10,
        n_glimpses=3
    )
    
    print(f"Model type: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    dummy_glimpses = torch.randn(batch_size, 3, 3, 16, 16)  # (batch, n_glimpses, channels, h, w)
    
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_glimpses)
    
    print(f"Input shape: {dummy_glimpses.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    
    # Test attention model
    attention_model = create_baseline_model(
        model_type='attention_weighted',
        num_classes=10,
        n_glimpses=3
    )
    
    attention_model.eval()
    with torch.no_grad():
        outputs, weights = attention_model(dummy_glimpses)
    
    print(f"\nAttention model:")
    print(f"Output shape: {outputs.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Attention weights example: {weights[0].numpy()}")
    
    print("Model creation demo completed!")


def demo_data_loading():
    """Demonstrate data loading with multi-glimpse support."""
    print("\n=== Data Loading Demo ===")
    
    try:
        # Create small dataset for demo
        train_loader, test_loader = create_baseline_dataloaders(
            dataset_name='cifar10',
            data_dir='./data',
            n_glimpses=3,
            batch_size=8,  # Small batch for demo
            num_workers=0  # Avoid multiprocessing issues
        )
        
        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Test dataset size: {len(test_loader.dataset)}")
        print(f"Batch size: {train_loader.batch_size}")
        
        # Get a sample batch
        glimpses, full_images, labels = next(iter(train_loader))
        
        print(f"\nSample batch:")
        print(f"Glimpses shape: {glimpses.shape}")
        print(f"Full images shape: {full_images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels.numpy()}")
        
        print("Data loading demo completed!")
        
    except Exception as e:
        print(f"Data loading failed: {e}")
        print("This might be due to network issues or disk space. You can skip this for now.")


def demo_metrics():
    """Demonstrate efficiency metrics calculation."""
    print("\n=== Metrics Demo ===")
    
    # Create metrics tracker
    metrics = EfficiencyMetrics()
    
    # Simulate some measurements
    np.random.seed(42)
    for i in range(10):
        accuracy = 0.7 + 0.2 * np.random.random()
        n_glimpses = 3
        processing_time = 0.1 + 0.05 * np.random.random()
        total_pixels = 3 * 16 * 16 * n_glimpses
        
        metrics.update(accuracy, n_glimpses, processing_time, total_pixels)
    
    # Get summary
    summary = metrics.get_summary()
    
    print("Sample efficiency metrics:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")
    
    print("Metrics demo completed!")


def quick_training_demo():
    """Run a very quick training demo (1 epoch)."""
    print("\n=== Quick Training Demo ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Create small dataset
        train_loader, test_loader = create_baseline_dataloaders(
            dataset_name='cifar10',
            data_dir='./data',
            n_glimpses=3,
            batch_size=32,
            num_workers=0
        )
        
        # Create model
        model = create_baseline_model(
            model_type='fixed',
            num_classes=10,
            n_glimpses=3
        ).to(device)
        
        # Simple training setup
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train for 1 epoch (just a few batches)
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        print("Running quick training (5 batches)...")
        for batch_idx, (glimpses, _, labels) in enumerate(train_loader):
            if batch_idx >= 5:  # Only 5 batches for demo
                break
                
            glimpses = glimpses.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(glimpses)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            print(f"  Batch {batch_idx+1}: Loss = {loss.item():.4f}, Acc = {100.*correct/total:.1f}%")
        
        print(f"Demo training completed!")
        print(f"Average loss: {total_loss/5:.4f}")
        print(f"Accuracy: {100.*correct/total:.1f}%")
        print("Quick training demo completed!")
        
    except Exception as e:
        print(f"Training demo failed: {e}")
        print("This might be due to memory constraints. The core functionality still works.")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Phase 1 Setup and Demo')
    parser.add_argument('--skip_data', action='store_true', help='Skip data loading demo')
    parser.add_argument('--skip_training', action='store_true', help='Skip training demo')
    parser.add_argument('--quick', action='store_true', help='Run only essential demos')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose output')
    
    args = parser.parse_args()
    
    print("Foveal Vision Project - Phase 1 Setup and Demo")
    if args.debug:
        print("DEBUG MODE ENABLED")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n Environment check failed. Please fix the issues above.")
        return
    
    # Enable debug mode if requested
    if args.debug:
        print("\n Debug info:")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"NumPy version: {np.__version__}")
        print(f"Working directory: {os.getcwd()}")
        print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    # Run demos
    if not args.quick:
        demo_foveal_sampling()
        demo_model_creation()
        
        if not args.skip_data:
            demo_data_loading()
        
        demo_metrics()
        
        if not args.skip_training:
            quick_training_demo()
    else:
        # Quick mode - just essential demos
        demo_model_creation()
        demo_metrics()
    
    print("\n Phase 1 setup and demo completed successfully!")


if __name__ == "__main__":
    main()