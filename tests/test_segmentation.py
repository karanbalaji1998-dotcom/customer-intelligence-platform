import pandas as pd
import pytest
from src.segmentation import assign_risk_value_segments

def test_all_four_segments():
    risk = pd.Series([0.9, 0.1, 0.9, 0.1], index=list("abcd"))
    clv = pd.Series([100.0, 100.0, 10.0, 10.0], index=list("abcd"))
    segments = assign_risk_value_segments(risk, clv, risk_threshold=0.5, high_value_threshold=50)
    assert segments["a"] == "High value / high risk"
    assert segments["b"] == "High value / low risk"
    assert segments["c"] == "Low value / high risk"
    assert segments["d"] == "Low value / low risk"

def test_invalid_probability_fails():
    with pytest.raises(ValueError):
        assign_risk_value_segments(pd.Series([1.2]), pd.Series([100.0]))
