"""Feature preparation helpers.

Keep target construction separate from predictor preparation to avoid leakage.
Fit transformations on training data only once a modelling pipeline is added.
"""
import pandas as pd

def split_target(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Return X and y after validating that the target column exists."""
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()
    return X, y

def drop_identifier_columns(
    X: pd.DataFrame, identifier_columns: list[str] | tuple[str, ...]
) -> pd.DataFrame:
    """Drop identifier columns from predictors if present."""
    return X.drop(columns=list(identifier_columns), errors="ignore").copy()
