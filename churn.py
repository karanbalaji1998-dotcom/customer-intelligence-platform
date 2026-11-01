"""Churn model training and evaluation placeholders.

Implement a leakage-safe sklearn Pipeline after the dataset schema is chosen.
"""
from typing import Any

def build_churn_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Any:
    """Build a preprocessing + classifier pipeline.

    TODO:
    - Use imputers and one-hot encoding for categorical fields.
    - Fit all preprocessing on training data only.
    - Compare a baseline with a tree-based model.
    """
    raise NotImplementedError(
        "Churn pipeline is not implemented yet. Complete this after selecting a dataset."
    )

def evaluate_churn_model(y_true, y_probability, threshold: float = 0.5) -> dict:
    """Return classification metrics from true labels and churn probabilities.

    TODO: implement ROC-AUC, PR-AUC, precision, recall, F1, and confusion matrix.
    """
    raise NotImplementedError("Churn evaluation is not implemented yet.")
