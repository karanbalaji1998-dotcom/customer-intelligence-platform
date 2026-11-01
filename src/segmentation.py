"""Customer risk/value segmentation."""
import pandas as pd

def assign_risk_value_segments(
    churn_probability: pd.Series,
    estimated_clv: pd.Series,
    risk_threshold: float = 0.5,
    high_value_threshold: float | None = None,
) -> pd.Series:
    """Assign transparent high/low risk × high/low value labels.

    If high_value_threshold is omitted, use the median CLV in the supplied
    population. In production, consider setting thresholds on a reference
    population and versioning them to avoid unexpected changes over time.
    """
    if not 0 <= risk_threshold <= 1:
        raise ValueError("risk_threshold must be between 0 and 1")
    risk = pd.Series(churn_probability, copy=True, dtype="float64")
    value = pd.Series(estimated_clv, copy=True, dtype="float64").reindex(risk.index)
    if risk.isna().any() or ((risk < 0) | (risk > 1)).any():
        raise ValueError("churn probabilities must be between 0 and 1")
    if value.isna().any() or (value < 0).any():
        raise ValueError("CLV must be non-missing and non-negative")
    if high_value_threshold is None:
        high_value_threshold = float(value.median())
    high_risk = risk >= risk_threshold
    high_value = value >= high_value_threshold

    labels = pd.Series(index=risk.index, dtype="object")
    labels[high_risk & high_value] = "High value / high risk"
    labels[~high_risk & high_value] = "High value / low risk"
    labels[high_risk & ~high_value] = "Low value / high risk"
    labels[~high_risk & ~high_value] = "Low value / low risk"
    return labels.rename("risk_value_segment")
