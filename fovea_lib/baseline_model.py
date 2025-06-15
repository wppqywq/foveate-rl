"""
Fixed Attention Baseline Model
Implements baseline model with fixed glimpse locations for foveal vision experiments.
Uses shared-weight ResNet-18 encoder with feature concatenation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple


class SharedResNetEncoder(nn.Module):
    """
    Shared-weight ResNet encoder for processing glimpses.
    Based on ResNet-18 architecture with modifications for small glimpse inputs.
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 feature_dim: int = 512,
                 pretrained: bool = False):
        """
        Initialize shared ResNet encoder.
        
        Args:
            input_channels: Number of input channels
            feature_dim: Dimension of output features
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Load ResNet-18 backbone - updated to avoid deprecation warning
        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)
        
        # Modify first conv layer if needed
        if input_channels != 3:
            resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, 
                                   stride=2, padding=3, bias=False)
        
        # Remove final FC layer and avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Add adaptive pooling and custom FC layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            features: Feature tensor of shape (batch_size, feature_dim)
        """
        features = self.backbone(x)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        features = self.dropout(features)
        features = self.fc(features)
        
        return features


class FixedAttentionModel(nn.Module):
    """
    Fixed attention baseline model for foveal vision.
    Processes multiple fixed glimpses and combines them for classification.
    """
    
    def __init__(self,
                 num_classes: int = 10,
                 n_glimpses: int = 3,
                 glimpse_size: int = 32,
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 pretrained_backbone: bool = False):
        """
        Initialize fixed attention model.
        
        Args:
            num_classes: Number of output classes
            n_glimpses: Number of glimpses per image
            glimpse_size: Size of each glimpse
            feature_dim: Dimension of glimpse features
            hidden_dim: Hidden dimension for classification head
            pretrained_backbone: Whether to use pretrained ResNet backbone
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.n_glimpses = n_glimpses
        self.feature_dim = feature_dim
        
        # Shared encoder for all glimpses
        self.glimpse_encoder = SharedResNetEncoder(
            input_channels=3,
            feature_dim=feature_dim,
            pretrained=pretrained_backbone
        )
        
        # Feature fusion and classification
        combined_feature_dim = feature_dim * n_glimpses
        self.classifier = nn.Sequential(
            nn.Linear(combined_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, glimpses: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            glimpses: Tensor of shape (batch_size, n_glimpses, channels, height, width)
            
        Returns:
            logits: Classification logits of shape (batch_size, num_classes)
        """
        batch_size, n_glimpses, channels, height, width = glimpses.shape
        
        # Reshape to process all glimpses together
        glimpses_flat = glimpses.view(-1, channels, height, width)
        
        # Extract features from all glimpses
        glimpse_features = self.glimpse_encoder(glimpses_flat)
        
        # Reshape back to separate glimpses
        glimpse_features = glimpse_features.view(batch_size, n_glimpses, self.feature_dim)
        
        # Concatenate features from all glimpses
        combined_features = glimpse_features.view(batch_size, -1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        return logits
    
    def get_glimpse_features(self, glimpses: torch.Tensor) -> torch.Tensor:
        """
        Extract features for each glimpse separately (for analysis).
        
        Args:
            glimpses: Tensor of shape (batch_size, n_glimpses, channels, height, width)
            
        Returns:
            features: Tensor of shape (batch_size, n_glimpses, feature_dim)
        """
        batch_size, n_glimpses, channels, height, width = glimpses.shape
        
        # Reshape to process all glimpses together
        glimpses_flat = glimpses.view(-1, channels, height, width)
        
        # Extract features
        glimpse_features = self.glimpse_encoder(glimpses_flat)
        
        # Reshape back to separate glimpses
        glimpse_features = glimpse_features.view(batch_size, n_glimpses, self.feature_dim)
        
        return glimpse_features


class AttentionWeightedModel(nn.Module):
    """
    Extension of baseline model with learned attention weights.
    Learns to weight glimpse features based on their importance.
    """
    
    def __init__(self,
                 num_classes: int = 10,
                 n_glimpses: int = 3,
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 pretrained_backbone: bool = False):
        """Initialize attention-weighted model."""
        super().__init__()
        
        self.num_classes = num_classes
        self.n_glimpses = n_glimpses
        self.feature_dim = feature_dim
        
        # Shared encoder
        self.glimpse_encoder = SharedResNetEncoder(
            input_channels=3,
            feature_dim=feature_dim,
            pretrained=pretrained_backbone
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, glimpses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention weighting.
        
        Args:
            glimpses: Tensor of shape (batch_size, n_glimpses, channels, height, width)
            
        Returns:
            logits: Classification logits
            attention_weights: Attention weights for each glimpse
        """
        batch_size, n_glimpses, channels, height, width = glimpses.shape
        
        # Extract glimpse features
        glimpses_flat = glimpses.view(-1, channels, height, width)
        glimpse_features = self.glimpse_encoder(glimpses_flat)
        glimpse_features = glimpse_features.view(batch_size, n_glimpses, self.feature_dim)
        
        # Compute attention weights
        attention_scores = self.attention(glimpse_features)  # (batch_size, n_glimpses, 1)
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=1)  # (batch_size, n_glimpses)
        
        # Weighted combination of features
        weighted_features = torch.sum(
            glimpse_features * attention_weights.unsqueeze(-1), 
            dim=1
        )  # (batch_size, feature_dim)
        
        # Classification
        logits = self.classifier(weighted_features)
        
        return logits, attention_weights


def create_baseline_model(model_type: str = 'fixed',
                         num_classes: int = 10,
                         n_glimpses: int = 3,
                         **kwargs) -> nn.Module:
    """
    Factory function to create baseline models.
    
    Args:
        model_type: Type of model ('fixed' or 'attention_weighted')
        num_classes: Number of output classes
        n_glimpses: Number of glimpses per image
        **kwargs: Additional model parameters
        
    Returns:
        Initialized model
    """
    if model_type == 'fixed':
        return FixedAttentionModel(
            num_classes=num_classes,
            n_glimpses=n_glimpses,
            **kwargs
        )
    elif model_type == 'attention_weighted':
        return AttentionWeightedModel(
            num_classes=num_classes,
            n_glimpses=n_glimpses,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class FullResolutionBaseline(nn.Module):
    """
    Full resolution baseline for comparison.
    Standard ResNet-18 processing full images.
    """
    
    def __init__(self, 
                 num_classes: int = 10,
                 pretrained: bool = False):
        """Initialize full resolution baseline."""
        super().__init__()
        
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet."""
        return self.resnet(x)


if __name__ == "__main__":
    # Test model creation
    model = create_baseline_model('fixed', num_classes=10, n_glimpses=3)
    
    # Test forward pass
    batch_size = 4
    n_glimpses = 3
    channels = 3
    height = width = 32
    
    dummy_input = torch.randn(batch_size, n_glimpses, channels, height, width)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test attention weighted model
    attention_model = create_baseline_model('attention_weighted', num_classes=10, n_glimpses=3)
    output, weights = attention_model(dummy_input)
    
    print(f"Attention model output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Attention model parameters: {sum(p.numel() for p in attention_model.parameters()):,}")