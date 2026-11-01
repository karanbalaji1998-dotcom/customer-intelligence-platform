"""Lightweight data and prediction monitoring placeholders."""
import pandas as pd

def numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calculate PSI using reference quantile bins.

    This basic implementation is intended as a starting point. Review binning,
    zero-count smoothing, and alert thresholds before production use.
    """
    if bins < 2:
        raise ValueError("bins must be at least 2")
    ref = pd.Series(reference).dropna().astype(float)
    cur = pd.Series(current).dropna().astype(float)
    if ref.empty or cur.empty:
        raise ValueError("reference and current data must contain numeric values")

    quantiles = sorted(set(ref.quantile([i / bins for i in range(bins + 1)]).tolist()))
    if len(quantiles) < 2:
        return 0.0
    edges = [-float("inf"), *quantiles[1:-1], float("inf")]
    ref_counts = pd.cut(ref, bins=edges, include_lowest=True).value_counts(sort=False)
    cur_counts = pd.cut(cur, bins=edges, include_lowest=True).value_counts(sort=False)
    ref_pct = (ref_counts / len(ref)).clip(lower=1e-6)
    cur_pct = (cur_counts / len(cur)).clip(lower=1e-6)
    return float(((cur_pct - ref_pct) * (cur_pct / ref_pct).apply(__import__("math").log)).sum())
