
"""Customer lifetime value (CLV) estimation helpers."""

import numpy as np
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
    - monthly_revenue is estimated revenue per customer per month.
    - monthly_retention_probability is the probability of remaining
      active from one month to the next.
    - gross_margin must be between 0 and 1, inclusive.
    - monthly_discount_rate is a non-negative monthly discount rate.

    This is a transparent fixed-horizon estimate, not a learned
    customer lifetime or churn model.
    """
    if (
        isinstance(horizon_months, bool)
        or not isinstance(horizon_months, (int, np.integer))
        or horizon_months < 1
    ):
        raise ValueError("horizon_months must be a positive integer")

    if not np.isfinite(gross_margin) or not 0 <= gross_margin <= 1:
        raise ValueError("gross_margin must be between 0 and 1")

    if (
        not np.isfinite(monthly_discount_rate)
        or monthly_discount_rate < 0
    ):
        raise ValueError(
            "monthly_discount_rate must be finite and non-negative"
        )

    revenue = pd.Series(
        monthly_revenue, copy=True, dtype="float64"
    )

    if not np.isfinite(revenue).all():
        raise ValueError(
            "monthly_revenue must contain only finite values"
        )

    if (revenue < 0).any():
        raise ValueError("monthly_revenue cannot be negative")

    if isinstance(monthly_retention_probability, pd.Series):
        retention = monthly_retention_probability.reindex(
            revenue.index
        ).astype("float64")
    else:
        retention = pd.Series(
            monthly_retention_probability,
            index=revenue.index,
            dtype="float64",
        )

    if (
        not np.isfinite(retention).all()
        or ((retention < 0) | (retention > 1)).any()
    ):
        raise ValueError(
            "retention probabilities must be finite "
            "and between 0 and 1"
        )

    clv = pd.Series(0.0, index=revenue.index)

    for month in range(1, horizon_months + 1):
        survival_probability = retention ** month
        discount_factor = (
            1 + monthly_discount_rate
        ) ** month

        clv += (
            revenue
            * gross_margin
            * survival_probability
            / discount_factor
        )

    return clv.rename("estimated_clv")
