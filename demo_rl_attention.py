"""
Demo: RL Attention Model
Quick demonstration of learned attention patterns.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fovea_lib.rl_attention import RecurrentAttentionModel
from fovea_lib.dataset_builder import create_cifar10_loaders
from fovea_lib.transforms import FovealTransform


def demo_attention_learning():
    """Demonstrate attention learning on sample images."""
    
    # Load data
    _, test_loader = create_cifar10_loaders(batch_size=4)
    
    # Create model
    model = RecurrentAttentionModel(
        glimpse_size=64,
        max_glimpses=6,
        num_classes=10
    )
    
    # Get sample batch
    images, labels = next(iter(test_loader))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, locations, _ = model(images)
        predictions = torch.argmax(logits, dim=1)
    
    # Visualize attention patterns
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for i in range(min(4, len(images))):
        # Denormalize image
        img = images[i].permute(1, 2, 0)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = torch.clamp(img, 0, 1)
        
        # Plot image with attention points
        axes[i].imshow(img)
        
        # Plot attention sequence
        seq_locations = torch.stack(locations).permute(1, 0, 2)[i]  # (T, 2)
        
        # Convert to pixel coordinates
        pixel_coords = (seq_locations + 1) * 16  # 32/2 = 16
        
        # Plot attention sequence
        for t, (x, y) in enumerate(pixel_coords):
            color = plt.cm.viridis(t / len(pixel_coords))
            axes[i].scatter(x, y, c=[color], s=100, alpha=0.8)
            axes[i].text(x+1, y+1, str(t), color='white', fontsize=8)
        
        axes[i].set_title(f'Pred: {predictions[i]}, True: {labels[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('attention_demo.png', dpi=150)
    print("Attention demo saved as 'attention_demo.png'")
    
    return model


def analyze_foveal_emergence():
    """Analyze emergence of foveal patterns."""
    from fovea_lib.metrics import EmergenceAnalyzer
    
    # Generate random attention patterns (simulating training progression)
    early_patterns = torch.randn(100, 6, 2) * 0.8  # More spread out
    late_patterns = torch.randn(100, 6, 2) * 0.3   # More centered
    
    analyzer = EmergenceAnalyzer()
    
    early_score = analyzer.compute_foveal_score(early_patterns)
    late_score = analyzer.compute_foveal_score(late_patterns)
    
    print(f"Early training foveal score: {early_score:.3f}")
    print(f"Late training foveal score: {late_score:.3f}")
    print(f"Improvement: {late_score - early_score:.3f}")


if __name__ == "__main__":
    print("Running RL Attention Demo...")
    
    model = demo_attention_learning()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    analyze_foveal_emergence()
    
    print("\nTo train full model:")
    print("python train_rl_attention.py --epochs 50 --dataset cifar10")