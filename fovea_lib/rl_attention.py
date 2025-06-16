"""
Reinforcement Learning Attention Model
Learns where to attend using REINFORCE algorithm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
from .transforms import GlimpseNetwork


class LocationNetwork(nn.Module):
    """Learns to predict attention locations."""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # (x, y) coordinates
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Output location in [-1, 1] range."""
        return torch.tanh(self.fc(features))


class RecurrentAttentionModel(nn.Module):
    """
    Recurrent attention model with learned location policy.
    Based on Mnih et al. 2014 and Cheung et al. 2016.
    """
    
    def __init__(self, 
                 glimpse_size: int = 64,
                 feature_dim: int = 256,
                 hidden_dim: int = 256,
                 num_classes: int = 10,
                 max_glimpses: int = 6):
        super().__init__()
        
        self.glimpse_size = glimpse_size
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_glimpses = max_glimpses
        
        # Core networks
        self.glimpse_network = GlimpseNetwork(glimpse_size, feature_dim)
        self.location_network = LocationNetwork(hidden_dim, hidden_dim)
        
        # Recurrent core
        self.rnn = nn.LSTMCell(feature_dim, hidden_dim)
        
        # Output heads
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.baseline = nn.Linear(hidden_dim, 1)  # Baseline for variance reduction
        
    def forward(self, 
                image: torch.Tensor,
                num_glimpses: Optional[int] = None) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass with attention sequence.
        
        Args:
            image: Input image tensor (B, C, H, W)
            num_glimpses: Number of glimpses (default: max_glimpses)
            
        Returns:
            logits: Classification logits
            locations: List of location coordinates
            log_probs: List of location log probabilities
        """
        if num_glimpses is None:
            num_glimpses = self.max_glimpses
            
        batch_size = image.size(0)
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_dim, device=image.device)
        c = torch.zeros(batch_size, self.hidden_dim, device=image.device)
        
        locations = []
        log_probs = []
        
        # Initial location (center)
        location = torch.zeros(batch_size, 2, device=image.device)
        
        for step in range(num_glimpses):
            # Extract glimpse at current location
            glimpse = self.extract_glimpse(image, location)
            
            # Encode glimpse
            glimpse_features = self.glimpse_network(glimpse)
            
            # Update RNN state
            h, c = self.rnn(glimpse_features, (h, c))
            
            # Store current location
            locations.append(location)
            
            # Predict next location (except for last step)
            if step < num_glimpses - 1:
                location_mean = self.location_network(h)
                
                # Sample location with exploration noise
                if self.training:
                    location_std = torch.ones_like(location_mean) * 0.1
                    location_dist = torch.distributions.Normal(location_mean, location_std)
                    location = location_dist.sample()
                    log_prob = location_dist.log_prob(location).sum(dim=1)
                    log_probs.append(log_prob)
                else:
                    location = location_mean
                    log_probs.append(torch.zeros(batch_size, device=image.device))
                
                # Clamp to valid range
                location = torch.clamp(location, -1, 1)
        
        # Final classification
        logits = self.classifier(h)
        
        return logits, locations, log_probs
    
    def extract_glimpse(self, image: torch.Tensor, location: torch.Tensor) -> torch.Tensor:
        """Extract foveated glimpse at given location."""
        batch_size = image.size(0)
        glimpses = []
        
        for i in range(batch_size):
            # Convert normalized coordinates to pixel coordinates
            img_h, img_w = image.shape[2:]
            x = int((location[i, 0].item() + 1) * img_w / 2)
            y = int((location[i, 1].item() + 1) * img_h / 2)
            
            # Extract and convert single image
            img_np = image[i].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 0.225 + 0.485) * 255  # Denormalize
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            # Create foveal sample
            from .transforms import FovealTransform
            transform = FovealTransform()
            glimpse = transform.create_glimpse_tensor(img_np, (x, y), self.glimpse_size)
            glimpses.append(glimpse)
        
        return torch.stack(glimpses).to(image.device)


class REINFORCETrainer:
    """Trainer for attention model using REINFORCE algorithm."""
    
    def __init__(self, 
                 model: RecurrentAttentionModel,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        self.classification_criterion = nn.CrossEntropyLoss()
        
    def train_step(self, images: torch.Tensor, labels: torch.Tensor) -> dict:
        """Single training step with REINFORCE."""
        self.model.train()
        
        # Forward pass
        logits, locations, log_probs = self.model(images)
        
        # Classification loss
        classification_loss = self.classification_criterion(logits, labels)
        
        # Compute reward (accuracy-based)
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == labels).float()
            reward = correct - 0.5  # Center around 0
        
        # REINFORCE loss
        reinforce_loss = 0
        if log_probs:
            for log_prob in log_probs:
                reinforce_loss = reinforce_loss - (log_prob * reward).mean()
        
        # Total loss
        total_loss = classification_loss + 0.1 * reinforce_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Statistics
        accuracy = correct.mean().item()
        
        return {
            'total_loss': total_loss.item(),
            'classification_loss': classification_loss.item(),
            'reinforce_loss': reinforce_loss.item() if isinstance(reinforce_loss, torch.Tensor) else 0,
            'accuracy': accuracy,
            'avg_reward': reward.mean().item()
        }
    
    def evaluate(self, dataloader) -> dict:
        """Evaluate model performance."""
        self.model.eval()
        
        total_correct = 0
        total_samples = 0
        attention_patterns = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits, locations, _ = self.model(images)
                predictions = torch.argmax(logits, dim=1)
                
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
                # Store attention patterns for analysis
                batch_patterns = torch.stack(locations).permute(1, 0, 2)  # (B, T, 2)
                attention_patterns.append(batch_patterns.cpu())
        
        accuracy = total_correct / total_samples
        attention_patterns = torch.cat(attention_patterns, dim=0)
        
        return {
            'accuracy': accuracy,
            'attention_patterns': attention_patterns
        }