"""
BetFred CV Service Models Module
Provides organized imports for classification models
"""
from .efficientnet_classifier import EfficientNetClassifier
from .densenet_classifier import DenseNetClassifier

__all__ = [
    'EfficientNetClassifier',
    'DenseNetClassifier'
]
