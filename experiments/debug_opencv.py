#!/usr/bin/env python3


import numpy as np
import cv2
import sys

def test_opencv_basic():
    """Test basic OpenCV functionality."""
    print("=== OpenCV Basic Test ===")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    # Create a simple test image
    test_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    print(f"Test image shape: {test_img.shape}")
    print(f"Test image dtype: {test_img.dtype}")
    print(f"Test image flags: {test_img.flags}")
    
    # Test resize
    try:
        resized = cv2.resize(test_img, (8, 8), interpolation=cv2.INTER_AREA)
        print(f"✅ OpenCV resize successful!")
        print(f"Resized shape: {resized.shape}")
        return True
    except Exception as e:
        print(f"❌ OpenCV resize failed: {e}")
        return False

def test_cifar_image():
    """Test with actual CIFAR image."""
    print("\n=== CIFAR Image Test ===")
    
    try:
        import torch
        from torchvision.datasets import CIFAR10
        from torchvision import transforms
        
        # Load CIFAR-10
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = CIFAR10(root='./data', train=False, download=False, transform=transform)
        
        # Get sample
        image_tensor, label = dataset[42]
        print(f"Tensor shape: {image_tensor.shape}")
        print(f"Tensor dtype: {image_tensor.dtype}")
        print(f"Tensor range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
        
        # Convert to numpy
        image_np = image_tensor.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        print(f"NumPy shape: {image_np.shape}")
        print(f"NumPy dtype: {image_np.dtype}")
        print(f"NumPy range: [{image_np.min()}, {image_np.max()}]")
        print(f"NumPy flags: {image_np.flags}")
        
        # Make contiguous
        image_np = np.ascontiguousarray(image_np)
        print(f"After ascontiguousarray: {image_np.flags}")
        
        # Test resize
        try:
            resized = cv2.resize(image_np, (8, 8), interpolation=cv2.INTER_AREA)
            print(f"✅ CIFAR image resize successful!")
            print(f"Resized shape: {resized.shape}")
            return True
        except Exception as e:
            print(f"❌ CIFAR image resize failed: {e}")
            print(f"Error type: {type(e)}")
            
            # Try different approaches
            print("\nTrying alternative approaches...")
            
            # Try different data types
            try:
                image_float = image_np.astype(np.float32) / 255.0
                resized_float = cv2.resize(image_float, (8, 8), interpolation=cv2.INTER_AREA)
                resized = (resized_float * 255).astype(np.uint8)
                print("✅ Float32 conversion worked!")
                return True
            except Exception as e2:
                print(f"❌ Float32 approach failed: {e2}")
            
            # Try copying
            try:
                image_copy = image_np.copy()
                resized = cv2.resize(image_copy, (8, 8), interpolation=cv2.INTER_AREA)
                print("✅ Copy approach worked!")
                return True
            except Exception as e3:
                print(f"❌ Copy approach failed: {e3}")
                
            return False
            
    except Exception as e:
        print(f"❌ CIFAR test setup failed: {e}")
        return False

def test_pil_fallback():
    """Test PIL as fallback."""
    print("\n=== PIL Fallback Test ===")
    
    try:
        from PIL import Image
        
        # Create test image
        test_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        
        # Convert to PIL and resize
        pil_img = Image.fromarray(test_img, 'RGB')
        pil_resized = pil_img.resize((8, 8), Image.LANCZOS)
        resized = np.array(pil_resized)
        
        print(f"✅ PIL resize successful!")
        print(f"Original shape: {test_img.shape}")
        print(f"Resized shape: {resized.shape}")
        return True
        
    except Exception as e:
        print(f"❌ PIL resize failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔍 OpenCV Compatibility Debug Script")
    print("=" * 40)
    
    results = []
    
    # Test basic OpenCV
    results.append(test_opencv_basic())
    
    # Test with CIFAR image
    results.append(test_cifar_image())
    
    # Test PIL fallback
    results.append(test_pil_fallback())
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    test_names = ["Basic OpenCV", "CIFAR Image", "PIL Fallback"]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    if any(results):
        print("\n✅ At least one method works - the system should function!")
    else:
        print("\n❌ All methods failed - there may be a deeper compatibility issue.")
        print("\nTroubleshooting suggestions:")
        print("1. Try: pip install opencv-python --upgrade")
        print("2. Try: pip uninstall opencv-python && pip install opencv-python-headless")
        print("3. Check if you're in a virtual environment")
        print("4. Try running with --quick flag to skip OpenCV-dependent demos")

if __name__ == "__main__":
    main()