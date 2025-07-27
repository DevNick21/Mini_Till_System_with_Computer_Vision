"""
Ensemble handwriting classifier combining multiple model architectures
"""
import torch
import torch.nn as nn
from .resnet_classifier import ResNetClassifier
from .efficientnet_classifier import EfficientNetClassifier
from .densenet_classifier import DenseNetClassifier
from ..utils.config import ALL_WRITERS, MODEL_SAVE_PATH
import os


class EnsembleHandwritingClassifier(nn.Module):
    """
    Ensemble of multiple classification models for handwriting identification

    Architecture:
    - Combines ResNet18, EfficientNet-B0, and DenseNet121 models
    - Uses weighted voting to make the final prediction
    - Improves robustness and accuracy over single model approaches
    """

    def __init__(self, num_writers=None, model_weights=None):
        super().__init__()

        # Use config defaults if not specified
        if num_writers is None:
            num_writers = len(ALL_WRITERS)

        self.num_writers = num_writers

        # Create individual models
        self.resnet_model = ResNetClassifier(num_writers=num_writers)
        self.efficientnet_model = EfficientNetClassifier(
            num_writers=num_writers)
        self.densenet_model = DenseNetClassifier(num_writers=num_writers)

        # Model weights for ensemble voting (can be learned or set manually)
        if model_weights is None:
            # Default weights - can be tuned based on validation performance
            self.model_weights = nn.Parameter(
                torch.tensor([0.4, 0.3, 0.3]), requires_grad=False)
        else:
            self.model_weights = nn.Parameter(
                torch.tensor(model_weights), requires_grad=False)

        # Normalize weights to sum to 1
        self.model_weights = nn.Parameter(
            self.model_weights / self.model_weights.sum())

        # Print initialization information
        print(f"EnsembleHandwritingClassifier initialized:")
        print(f"  Writers: {num_writers}")
        print(f"  Models: ResNet18, EfficientNet-B0, DenseNet121")
        print(f"  Model weights: {self.model_weights.tolist()}")
        print(
            f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def load_individual_models(self):
        """Load pre-trained weights for individual models"""
        models = {
            "resnet": self.resnet_model,
            "efficientnet": self.efficientnet_model,
            "densenet": self.densenet_model
        }

        # Try to load each model's weights
        for model_name, model in models.items():
            model_path = os.path.join(
                MODEL_SAVE_PATH, f"best_{model_name}_classifier.pth")

            if os.path.exists(model_path):
                try:
                    model.load_state_dict(torch.load(
                        model_path, map_location="cpu"))
                    print(f"  ✓ Loaded {model_name} weights from {model_path}")
                except Exception as e:
                    print(f"  ✗ Error loading {model_name} weights: {e}")
            else:
                # Fall back to the general weights file with renamed parameters
                try:
                    general_path = os.path.join(
                        MODEL_SAVE_PATH, "best_handwriting_classifier.pth")
                    if os.path.exists(general_path) and model_name == "resnet":
                        model.load_state_dict(torch.load(
                            general_path, map_location="cpu"))
                        print(
                            f"  ✓ Loaded {model_name} weights from general model file")
                    else:
                        print(
                            f"  ✗ No pre-trained weights found for {model_name}")
                except Exception as e:
                    print(
                        f"  ✗ Error loading general weights for {model_name}: {e}")

    def forward(self, x):
        """Forward pass through the ensemble network"""
        # Get logits from each model
        resnet_logits = self.resnet_model(x)
        efficientnet_logits = self.efficientnet_model(x)
        densenet_logits = self.densenet_model(x)

        # Convert logits to probabilities
        resnet_probs = torch.softmax(resnet_logits, dim=1)
        efficientnet_probs = torch.softmax(efficientnet_logits, dim=1)
        densenet_probs = torch.softmax(densenet_logits, dim=1)

        # Weight the probabilities by model weights
        weighted_probs = (
            resnet_probs * self.model_weights[0] +
            efficientnet_probs * self.model_weights[1] +
            densenet_probs * self.model_weights[2]
        )

        # Convert back to logits for loss calculation compatibility
        # Using a small epsilon to avoid log(0)
        epsilon = 1e-8
        weighted_logits = torch.log(weighted_probs + epsilon)

        return weighted_logits

    def predict_proba(self, x):
        """Get prediction probabilities from ensemble"""
        logits = self.forward(x)
        # No need to apply softmax again as we're already working with probabilities
        return torch.exp(logits)

    def predict(self, x):
        """Get prediction and confidence from ensemble"""
        probabilities = self.predict_proba(x)
        confidence, predicted = torch.max(probabilities, dim=1)
        return predicted, confidence

    def individual_predictions(self, x):
        """Get predictions from each individual model for analysis"""
        # ResNet predictions
        with torch.no_grad():
            resnet_probs = torch.softmax(self.resnet_model(x), dim=1)
            resnet_conf, resnet_pred = torch.max(resnet_probs, dim=1)

            efficientnet_probs = torch.softmax(
                self.efficientnet_model(x), dim=1)
            efficientnet_conf, efficientnet_pred = torch.max(
                efficientnet_probs, dim=1)

            densenet_probs = torch.softmax(self.densenet_model(x), dim=1)
            densenet_conf, densenet_pred = torch.max(densenet_probs, dim=1)

        return {
            "resnet": (resnet_pred, resnet_conf),
            "efficientnet": (efficientnet_pred, efficientnet_conf),
            "densenet": (densenet_pred, densenet_conf)
        }


# Test the model
if __name__ == "__main__":
    # Test model creation
    model = EnsembleHandwritingClassifier()
    model.load_individual_models()

    # Test with dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)
        probabilities = model.predict_proba(dummy_input)
        predicted, confidence = model.predict(dummy_input)
        individual_preds = model.individual_predictions(dummy_input)

    print(f"\nEnsemble model test successful:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Ensemble prediction: Writer {predicted.item()}")
    print(f"  Ensemble confidence: {confidence.item():.3f}")

    # Show individual model predictions
    for model_name, (pred, conf) in individual_preds.items():
        print(
            f"  {model_name.capitalize()} prediction: Writer {pred.item()} ({conf.item():.3f})")
