import pandas as pd
import pytest
from src.clv import estimate_simple_clv

def test_clv_zero_retention_is_zero():
    result = estimate_simple_clv(pd.Series([100.0]), 0.0, horizon_months=12)
    assert result.iloc[0] == 0.0

def test_clv_full_retention_without_discount():
    result = estimate_simple_clv(pd.Series([100.0]), 1.0, horizon_months=12, gross_margin=0.5)
    assert result.iloc[0] == pytest.approx(600.0)

def test_clv_rejects_invalid_margin():
    with pytest.raises(ValueError):
        estimate_simple_clv(pd.Series([100.0]), 0.9, gross_margin=1.5)
