"""
EfficientNet-based handwriting classifier model
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from config import ALL_WRITERS, DROPOUT_RATE


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

        # EfficientNet backbone
        self.used_pretrained = False
        try:
            if use_pretrained:
                backbone = efficientnet_b0(
                    weights=EfficientNet_B0_Weights.DEFAULT)
                self.used_pretrained = True
            else:
                backbone = efficientnet_b0(weights=None)
        except Exception as e:
            print(
                f"Warning: failed to load pretrained weights, falling back to random init. Reason: {e}"
            )
            backbone = efficientnet_b0(weights=None)
            self.used_pretrained = False

        # Use only convolutional feature extractor
        self.features = backbone.features

        # Adaptive pooling to get (batch, 1280, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),  # now 1280 features
            nn.Dropout(dropout_rate),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_writers),
        )

        # Diagnostics
        print("EfficientNetClassifier initialized:")
        print(f"  Writers: {num_writers}")
        print(f"  Dropout rate: {dropout_rate}")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(
            f"  Pretrained weights: {'YES' if self.used_pretrained else 'NO'}")

    def forward(self, x):
        """Forward pass"""
        features = self.features(x)
        features = self.pool(features)
        features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x):
        """Return prediction probabilities"""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1).cpu()

    def predict(self, x):
        """Return predicted class and confidence"""
        probabilities = self.predict_proba(x)
        confidence, predicted = torch.max(probabilities, dim=1)
        return predicted, confidence


if __name__ == "__main__":
    model = EfficientNetClassifier()
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
