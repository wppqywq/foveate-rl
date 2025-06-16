"""
Metrics for analyzing foveal attention emergence.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


class EmergenceAnalyzer:
    """Analyzes emergence of foveal patterns in learned attention."""
    
    def __init__(self, image_size: int = 32):
        self.image_size = image_size
        
    def compute_foveal_score(self, attention_patterns: torch.Tensor) -> float:
        """
        Compute how foveal the attention patterns are.
        
        Args:
            attention_patterns: (N, T, 2) tensor of attention sequences
            
        Returns:
            Foveal score (higher = more foveal)
        """
        scores = []
        
        for sequence in attention_patterns:
            # Convert to pixel coordinates
            locations = (sequence + 1) * self.image_size / 2
            
            # Compute center of mass
            center_x = locations[:, 0].mean()
            center_y = locations[:, 1].mean()
            
            # Distance from center of image
            img_center = self.image_size / 2
            center_distance = np.sqrt((center_x - img_center)**2 + (center_y - img_center)**2)
            
            # Compute spread around center of mass
            distances_from_com = torch.sqrt(
                (locations[:, 0] - center_x)**2 + (locations[:, 1] - center_y)**2
            )
            spread = distances_from_com.mean()
            
            # Foveal score: low center distance, low spread
            foveal_score = 1.0 / (1.0 + center_distance + spread)
            scores.append(foveal_score)
        
        return np.mean(scores)
    
    def compute_scanpath_similarity(self, 
                                  attention_patterns: torch.Tensor,
                                  reference_patterns: torch.Tensor = None) -> float:
        """Compute similarity between scanpaths."""
        if reference_patterns is None:
            # Self-similarity
            patterns = attention_patterns.numpy().reshape(len(attention_patterns), -1)
            distances = pdist(patterns, metric='euclidean')
            return 1.0 / (1.0 + np.mean(distances))
        else:
            # Similarity to reference
            patterns1 = attention_patterns.numpy().reshape(len(attention_patterns), -1)
            patterns2 = reference_patterns.numpy().reshape(len(reference_patterns), -1)
            distances = []
            
            for p1 in patterns1:
                for p2 in patterns2:
                    distances.append(np.linalg.norm(p1 - p2))
            
            return 1.0 / (1.0 + np.mean(distances))
    
    def analyze_temporal_patterns(self, attention_patterns: torch.Tensor) -> Dict[str, float]:
        """Analyze temporal properties of attention sequences."""
        results = {}
        
        # Average sequence length
        results['mean_sequence_length'] = attention_patterns.size(1)
        
        # Movement patterns
        movements = []
        for sequence in attention_patterns:
            diffs = sequence[1:] - sequence[:-1]
            movement_distances = torch.sqrt((diffs**2).sum(dim=1))
            movements.append(movement_distances.mean().item())
        
        results['mean_movement_distance'] = np.mean(movements)
        results['std_movement_distance'] = np.std(movements)
        
        # Fixation clustering
        clustering_scores = []
        for sequence in attention_patterns:
            locations = sequence.numpy()
            if len(locations) > 1:
                distances = pdist(locations)
                clustering_scores.append(1.0 / (1.0 + np.mean(distances)))
        
        results['clustering_score'] = np.mean(clustering_scores)
        
        return results


class TrainingMonitor:
    """Monitors training progress and emergence."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.epoch_metrics = []
        self.attention_histories = []
    
    def update(self, epoch: int, metrics: dict, attention_patterns: torch.Tensor):
        """Update with epoch results."""
        analyzer = EmergenceAnalyzer()
        
        emergence_metrics = {
            'epoch': epoch,
            'accuracy': metrics['accuracy'],
            'foveal_score': analyzer.compute_foveal_score(attention_patterns),
            **analyzer.analyze_temporal_patterns(attention_patterns)
        }
        
        self.epoch_metrics.append(emergence_metrics)
        self.attention_histories.append(attention_patterns.clone())
    
    def plot_emergence_curves(self, save_path: str = None):
        """Plot emergence of foveal patterns over training."""
        if not self.epoch_metrics:
            return
        
        epochs = [m['epoch'] for m in self.epoch_metrics]
        accuracies = [m['accuracy'] for m in self.epoch_metrics]
        foveal_scores = [m['foveal_score'] for m in self.epoch_metrics]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(epochs, accuracies, 'b-', label='Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Classification Performance')
        ax1.grid(True)
        
        ax2.plot(epochs, foveal_scores, 'r-', label='Foveal Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Foveal Score')
        ax2.set_title('Foveal Pattern Emergence')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def get_summary(self) -> dict:
        """Get training summary."""
        if not self.epoch_metrics:
            return {}
        
        final_metrics = self.epoch_metrics[-1]
        initial_metrics = self.epoch_metrics[0]
        
        return {
            'final_accuracy': final_metrics['accuracy'],
            'final_foveal_score': final_metrics['foveal_score'],
            'accuracy_improvement': final_metrics['accuracy'] - initial_metrics['accuracy'],
            'foveal_improvement': final_metrics['foveal_score'] - initial_metrics['foveal_score'],
            'total_epochs': len(self.epoch_metrics)
        }


def compute_attention_statistics(attention_patterns: torch.Tensor) -> dict:
    """Compute basic statistics for attention patterns."""
    stats = {}
    
    # Convert to numpy for easier computation
    patterns = attention_patterns.numpy()
    
    # Basic statistics
    stats['mean_x'] = np.mean(patterns[:, :, 0])
    stats['mean_y'] = np.mean(patterns[:, :, 1])
    stats['std_x'] = np.std(patterns[:, :, 0])
    stats['std_y'] = np.std(patterns[:, :, 1])
    
    # Coverage area
    x_range = np.max(patterns[:, :, 0]) - np.min(patterns[:, :, 0])
    y_range = np.max(patterns[:, :, 1]) - np.min(patterns[:, :, 1])
    stats['coverage_area'] = x_range * y_range
    
    # Sequence diversity
    unique_sequences = []
    for seq in patterns:
        seq_rounded = np.round(seq, 2)
        seq_tuple = tuple(map(tuple, seq_rounded))
        if seq_tuple not in unique_sequences:
            unique_sequences.append(seq_tuple)
    
    stats['sequence_diversity'] = len(unique_sequences) / len(patterns)
    
    return stats