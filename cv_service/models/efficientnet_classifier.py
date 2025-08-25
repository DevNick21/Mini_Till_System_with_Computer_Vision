"""
EfficientNet-based handwriting classifier model.
"""

from config import ALL_WRITERS, DROPOUT_RATE
import os
import sys
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# External configuration


class EfficientNetClassifier(nn.Module):
    """
    CNN-based handwriting classifier using EfficientNet-B0 backbone.

    Architecture:
    - EfficientNet-B0 backbone
    - Adaptive pooling to convert feature maps to fixed size
    - Custom classification head for writer identification
    - Dropout for regularization

    Input: (batch_size, 3, 224, 224)
    Output: (batch_size, num_writers)
    """

    def __init__(self, num_writers=None, dropout_rate=None, use_pretrained: bool = True):
        super().__init__()

        # Defaults from config
        if num_writers is None:
            num_writers = len(ALL_WRITERS)
        if dropout_rate is None:
            dropout_rate = DROPOUT_RATE

        self.num_writers = num_writers
        self.dropout_rate = dropout_rate

        # EfficientNet backbone
        used_pretrained = False
        try:
            if use_pretrained:
                backbone = efficientnet_b0(
                    weights=EfficientNet_B0_Weights.DEFAULT)
                used_pretrained = True
            else:
                backbone = efficientnet_b0(weights=None)
        except Exception as e:
            print(
                f"Warning: failed to load pretrained weights, falling back to random init. Reason: {e}"
            )
            backbone = efficientnet_b0(weights=None)
            used_pretrained = False

        # Use only convolutional feature extractor
        self.features = backbone.features

        # Adaptive pooling to get (batch, 1280, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classification head for writer ID
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1280, self.num_writers),
        )

        # Diagnostics
        print("EfficientNetClassifier initialized:")
        print(f"  Writers: {self.num_writers}")
        print(f"  Dropout rate: {self.dropout_rate}")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  Pretrained weights: {'YES' if used_pretrained else 'NO'}")

    def forward(self, x):
        """Forward pass"""
        features = self.features(x)
        features = self.pool(features)
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    model = EfficientNetClassifier()
    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    print(f"\nModel test successful:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Predicted writer ID: {predicted.item()}")
    print(f"  Confidence: {confidence.item():.3f}")
