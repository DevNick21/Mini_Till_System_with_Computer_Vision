#!/usr/bin/env python
"""
Run script for model evaluation that adds the current directory to Python path
"""
from src.utils.config import MODEL_SAVE_PATH, ALL_WRITERS
from src.training.evaluate_model import (
    evaluate_model_detailed,
    analyze_similar_writers,
    generate_betfred_visualizations
)
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import and run the evaluation module

if __name__ == "__main__":
    print("\n=== BetFred Handwriting Classification Evaluation ===")
    print("=== EfficientNet Model Evaluation ===\n")

    # Run the evaluation
    results, details = evaluate_model_detailed()
    confusion_pairs = analyze_similar_writers(details)

    # Prepare data for visualizations
    all_labels = [d['true_id'] for d in details]
    all_predictions = [d['predicted_id'] for d in details]
    all_confidences = [d['confidence'] for d in details]
    writer_performance = results['writer_performance']

    # Generate visualizations
    generate_betfred_visualizations(
        all_labels,
        all_predictions,
        all_confidences,
        writer_performance,
        save_dir=MODEL_SAVE_PATH,
        all_writers=ALL_WRITERS
    )

    print("\nEvaluation complete! Results saved to:", MODEL_SAVE_PATH)
