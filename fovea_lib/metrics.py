"""
Efficiency Metrics for Foveal Vision
Implements metrics to evaluate computational efficiency and information processing.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict


class EfficiencyMetrics:
    """
    Tracks and computes efficiency metrics for foveal vision models.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.accuracies = []
        self.n_glimpses_used = []
        self.processing_times = []
        self.pixel_counts = []
        self.flops_counts = []
        
    def update(self, 
               accuracy: float,
               n_glimpses: int,
               processing_time: float,
               total_pixels: int,
               flops: Optional[int] = None):
        """
        Update metrics with new measurements.
        
        Args:
            accuracy: Classification accuracy for this batch
            n_glimpses: Number of glimpses used
            processing_time: Time taken for processing (seconds)
            total_pixels: Total number of pixels processed
            flops: Number of floating point operations (optional)
        """
        self.accuracies.append(accuracy)
        self.n_glimpses_used.append(n_glimpses)
        self.processing_times.append(processing_time)
        self.pixel_counts.append(total_pixels)
        if flops is not None:
            self.flops_counts.append(flops)
    
    def compute_information_efficiency(self) -> float:
        """
        Compute Information Efficiency (IE) = accuracy / n_glimpses.
        
        Returns:
            Average information efficiency
        """
        if not self.accuracies:
            return 0.0
        
        ie_scores = [acc / n_glimpses for acc, n_glimpses 
                    in zip(self.accuracies, self.n_glimpses_used)]
        return np.mean(ie_scores)
    
    def compute_bandwidth_efficiency(self) -> float:
        """
        Compute Bandwidth Efficiency (BW) = total_pixels / n_glimpses.
        
        Returns:
            Average bandwidth efficiency (pixels per glimpse)
        """
        if not self.pixel_counts:
            return 0.0
        
        bw_scores = [pixels / n_glimpses for pixels, n_glimpses 
                    in zip(self.pixel_counts, self.n_glimpses_used)]
        return np.mean(bw_scores)
    
    def compute_temporal_efficiency(self) -> float:
        """
        Compute Temporal Efficiency = accuracy / processing_time.
        
        Returns:
            Average temporal efficiency (accuracy per second)
        """
        if not self.processing_times:
            return 0.0
        
        te_scores = [acc / time for acc, time 
                    in zip(self.accuracies, self.processing_times)]
        return np.mean(te_scores)
    
    def compute_computational_efficiency(self) -> float:
        """
        Compute Computational Efficiency = accuracy / FLOPs.
        
        Returns:
            Average computational efficiency (accuracy per FLOP)
        """
        if not self.flops_counts:
            return 0.0
        
        ce_scores = [acc / flops for acc, flops 
                    in zip(self.accuracies, self.flops_counts)]
        return np.mean(ce_scores)
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary of all efficiency metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        summary = {
            'mean_accuracy': np.mean(self.accuracies) if self.accuracies else 0.0,
            'mean_n_glimpses': np.mean(self.n_glimpses_used) if self.n_glimpses_used else 0.0,
            'information_efficiency': self.compute_information_efficiency(),
            'bandwidth_efficiency': self.compute_bandwidth_efficiency(),
            'temporal_efficiency': self.compute_temporal_efficiency(),
        }
        
        if self.flops_counts:
            summary['computational_efficiency'] = self.compute_computational_efficiency()
            summary['mean_flops'] = np.mean(self.flops_counts)
        
        if self.processing_times:
            summary['mean_processing_time'] = np.mean(self.processing_times)
        
        return summary


class ModelComparator:
    """
    Compares efficiency metrics between different models or strategies.
    """
    
    def __init__(self):
        self.model_metrics = defaultdict(EfficiencyMetrics)
    
    def add_model(self, model_name: str):
        """Add a new model to track."""
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = EfficiencyMetrics()
    
    def update_model(self, model_name: str, **kwargs):
        """Update metrics for a specific model."""
        self.model_metrics[model_name].update(**kwargs)
    
    def compare_models(self) -> Dict[str, Dict[str, float]]:
        """
        Compare all tracked models.
        
        Returns:
            Dictionary with model names as keys and metric summaries as values
        """
        comparison = {}
        for model_name, metrics in self.model_metrics.items():
            comparison[model_name] = metrics.get_summary()
        
        return comparison
    
    def compute_relative_improvement(self, 
                                   baseline_model: str, 
                                   comparison_model: str) -> Dict[str, float]:
        """
        Compute relative improvement of comparison_model over baseline_model.
        
        Args:
            baseline_model: Name of baseline model
            comparison_model: Name of model to compare
            
        Returns:
            Dictionary of relative improvements (positive = better)
        """
        baseline_summary = self.model_metrics[baseline_model].get_summary()
        comparison_summary = self.model_metrics[comparison_model].get_summary()
        
        improvements = {}
        for metric in baseline_summary:
            if baseline_summary[metric] != 0:
                relative_change = (comparison_summary[metric] - baseline_summary[metric]) / baseline_summary[metric]
                improvements[f"{metric}_improvement"] = relative_change * 100  # Convert to percentage
        
        return improvements


def measure_model_efficiency(model: torch.nn.Module,
                           dataloader: torch.utils.data.DataLoader,
                           device: torch.device,
                           max_batches: Optional[int] = None) -> Dict[str, float]:
    """
    Measure efficiency metrics for a given model and dataset.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader containing evaluation data
        device: Device to run evaluation on
        max_batches: Maximum number of batches to evaluate (None = all)
        
    Returns:
        Dictionary containing efficiency metrics
    """
    model.eval()
    metrics = EfficiencyMetrics()
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (glimpses, full_images, labels) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Move to device
            glimpses = glimpses.to(device)
            labels = labels.to(device)
            
            # Measure processing time
            start_time = time.time()
            
            # Forward pass
            outputs = model(glimpses)
            
            processing_time = time.time() - start_time
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            batch_correct = (predicted == labels).sum().item()
            batch_size = labels.size(0)
            batch_accuracy = batch_correct / batch_size
            
            total_correct += batch_correct
            total_samples += batch_size
            
            # Calculate metrics
            n_glimpses = glimpses.shape[1]  # Number of glimpses per image
            total_pixels = glimpses.numel()  # Total pixels processed
            
            metrics.update(
                accuracy=batch_accuracy,
                n_glimpses=n_glimpses,
                processing_time=processing_time,
                total_pixels=total_pixels
            )
    
    # Add overall accuracy
    overall_accuracy = total_correct / total_samples
    summary = metrics.get_summary()
    summary['overall_accuracy'] = overall_accuracy
    
    return summary


def compute_pixel_efficiency_curve(model: torch.nn.Module,
                                 dataloader: torch.utils.data.DataLoader,
                                 device: torch.device,
                                 glimpse_counts: List[int] = [1, 2, 3, 4, 5]) -> Dict[int, float]:
    """
    Compute accuracy vs number of glimpses curve.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader containing evaluation data
        device: Device to run evaluation on
        glimpse_counts: List of glimpse counts to evaluate
        
    Returns:
        Dictionary mapping glimpse count to accuracy
    """
    model.eval()
    accuracy_curve = {}
    
    for n_glimpses in glimpse_counts:
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for glimpses, _, labels in dataloader:
                # Use only first n_glimpses
                glimpses = glimpses[:, :n_glimpses].to(device)
                labels = labels.to(device)
                
                outputs = model(glimpses)
                _, predicted = torch.max(outputs.data, 1)
                
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        accuracy_curve[n_glimpses] = total_correct / total_samples
    
    return accuracy_curve


class FLOPCounter:
    """
    Simple FLOP counter for PyTorch models.
    """
    
    def __init__(self):
        self.total_flops = 0
        self.hooks = []
    
    def conv_flop_count(self, module, input, output):
        """Count FLOPs for convolution layers."""
        batch_size = input[0].shape[0]
        output_dims = output.shape[2:]
        kernel_dims = module.kernel_size
        in_channels = module.in_channels
        out_channels = module.out_channels
        groups = module.groups
        
        filters_per_channel = out_channels // groups
        conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels // groups
        
        active_elements_count = batch_size * int(np.prod(output_dims))
        overall_conv_flops = conv_per_position_flops * active_elements_count * filters_per_channel
        
        self.total_flops += overall_conv_flops
    
    def linear_flop_count(self, module, input, output):
        """Count FLOPs for linear layers."""
        batch_size = input[0].shape[0]
        self.total_flops += batch_size * module.in_features * module.out_features
    
    def register_hooks(self, model):
        """Register hooks to count FLOPs."""
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                hook = module.register_forward_hook(self.conv_flop_count)
                self.hooks.append(hook)
            elif isinstance(module, torch.nn.Linear):
                hook = module.register_forward_hook(self.linear_flop_count)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def reset(self):
        """Reset FLOP counter."""
        self.total_flops = 0