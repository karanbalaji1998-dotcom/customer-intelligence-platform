"""Customer lifetime value (CLV) estimation helpers."""
import pandas as pd

def estimate_simple_clv(
    monthly_revenue: pd.Series,
    monthly_retention_probability: pd.Series | float,
    horizon_months: int = 12,
    gross_margin: float = 1.0,
    monthly_discount_rate: float = 0.0,
) -> pd.Series:
    """Estimate expected discounted gross profit over a fixed horizon.

    Assumptions:
    - monthly_revenue is a per-customer monthly revenue estimate.
    - retention_probability is the probability of remaining active each month.
    - gross_margin is between 0 and 1.
    - monthly_discount_rate is non-negative.
    This is a transparent fixed-horizon estimate, not a learned lifetime model.
    """
    if horizon_months < 1:
        raise ValueError("horizon_months must be at least 1")
    if not 0 <= gross_margin <= 1:
        raise ValueError("gross_margin must be between 0 and 1")
    if monthly_discount_rate < 0:
        raise ValueError("monthly_discount_rate must be non-negative")

    revenue = pd.Series(monthly_revenue, copy=True, dtype="float64")
    retention = pd.Series(monthly_retention_probability, index=revenue.index, dtype="float64") \
        if not isinstance(monthly_retention_probability, pd.Series) \
        else monthly_retention_probability.reindex(revenue.index).astype("float64")

    if (revenue < 0).any():
        raise ValueError("monthly_revenue cannot be negative")
    if retention.isna().any() or ((retention < 0) | (retention > 1)).any():
        raise ValueError("retention probabilities must be between 0 and 1")

    clv = pd.Series(0.0, index=revenue.index)
    for month in range(1, horizon_months + 1):
        survival = retention ** month
        discount = (1 + monthly_discount_rate) ** month
        clv += (revenue * gross_margin * survival / discount)
    return clv.rename("estimated_clv")
