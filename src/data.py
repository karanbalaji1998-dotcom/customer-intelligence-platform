"""Data loading and basic validation utilities."""
from pathlib import Path
import pandas as pd

def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file and raise a helpful error if it is missing or empty."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV not found: {file_path}")
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"CSV contains no rows: {file_path}")
    return df

def validate_required_columns(
    df: pd.DataFrame, required_columns: list[str] | tuple[str, ...]
) -> None:
    """Raise ValueError when required columns are missing."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def basic_data_quality_report(df: pd.DataFrame) -> dict:
    """Return lightweight quality checks without modifying the input data."""
    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "columns": list(df.columns),
    }
