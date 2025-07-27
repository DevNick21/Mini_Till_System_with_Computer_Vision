"""
DenseNet-based handwriting classifier model
"""
import torch
import torch.nn as nn
from torchvision.models import densenet121
from ..utils.config import ALL_WRITERS, DROPOUT_RATE


class DenseNetClassifier(nn.Module):
    """
    CNN-based handwriting classifier using DenseNet backbone

    Architecture:
    - DenseNet121 backbone (pre-trained on ImageNet)
    - Custom classification head for writer identification
    - With Dropout for regularization
    """

    def __init__(self, num_writers=None, dropout_rate=None):
        super().__init__()

        # Use config defaults if not specified
        if num_writers is None:
            num_writers = len(ALL_WRITERS)
        if dropout_rate is None:
            dropout_rate = DROPOUT_RATE

        self.num_writers = num_writers

        # Pre-trained DenseNet backbone (without final layer)
        backbone = densenet121(weights='DEFAULT')

        # Store features extractor
        self.features = backbone.features

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Custom classification head - DenseNet121 has 1024 features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_writers)
        )

        print(f"DenseNetClassifier initialized:")
        print(f"  Writers: {num_writers}")
        print(f"  Dropout rate: {dropout_rate}")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x):
        """Forward pass through the network"""
        # Extract features using DenseNet backbone
        features = self.features(x)

        # Apply global average pooling
        features = self.avg_pool(features)

        # Classify using custom head
        logits = self.classifier(features)

        return logits

    def predict_proba(self, x):
        """Get prediction probabilities"""
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities

    def predict(self, x):
        """Get prediction and confidence"""
        probabilities = self.predict_proba(x)
        confidence, predicted = torch.max(probabilities, 1)
        return predicted, confidence


# Test the model
if __name__ == "__main__":
    # Test model creation
    model = DenseNetClassifier()

    # Test with dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)
        probabilities = model.predict_proba(dummy_input)
        predicted, confidence = model.predict(dummy_input)

    print(f"\nModel test successful:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Predicted writer ID: {predicted.item()}")
    print(f"  Confidence: {confidence.item():.3f}")
