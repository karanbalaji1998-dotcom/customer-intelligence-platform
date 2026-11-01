
"""Tests for customer risk/value segmentation."""

import pandas as pd
import pytest

from src.segmentation import assign_risk_value_segments


def test_all_four_segments():
    risk = pd.Series(
        [0.9, 0.1, 0.9, 0.1],
        index=["a", "b", "c", "d"],
    )
    clv = pd.Series(
        [100.0, 100.0, 10.0, 10.0],
        index=["a", "b", "c", "d"],
    )

    segments = assign_risk_value_segments(
        risk,
        clv,
        risk_threshold=0.5,
        high_value_threshold=50.0,
    )

    assert segments["a"] == "High value / high risk"
    assert segments["b"] == "High value / low risk"
    assert segments["c"] == "Low value / high risk"
    assert segments["d"] == "Low value / low risk"


def test_invalid_probability_fails():
    with pytest.raises(ValueError):
        assign_risk_value_segments(
            pd.Series([1.2]),
            pd.Series([100.0]),
        )


def test_risk_threshold_boundary():
    risk = pd.Series(
        [0.5, 0.49],
        index=["boundary", "below"],
    )
    clv = pd.Series(
        [100.0, 100.0],
        index=["boundary", "below"],
    )

    segments = assign_risk_value_segments(
        risk,
        clv,
        risk_threshold=0.5,
        high_value_threshold=50.0,
    )

    assert segments["boundary"] == "High value / high risk"
    assert segments["below"] == "High value / low risk"


def test_rejects_negative_clv():
    with pytest.raises(ValueError):
        assign_risk_value_segments(
            pd.Series([0.8]),
            pd.Series([-10.0]),
        )


def test_rejects_missing_clv():
    with pytest.raises(ValueError):
        assign_risk_value_segments(
            pd.Series([0.8]),
            pd.Series([float("nan")]),
        )


def test_preserves_customer_indexes():
    risk = pd.Series(
        [0.9, 0.1],
        index=["customer_a", "customer_b"],
    )
    clv = pd.Series(
        [100.0, 10.0],
        index=["customer_a", "customer_b"],
    )

    segments = assign_risk_value_segments(
        risk,
        clv,
        risk_threshold=0.5,
        high_value_threshold=50.0,
    )

    assert list(segments.index) == [
        "customer_a",
        "customer_b",
    ]
    assert segments.name == "risk_value_segment"


def test_rejects_invalid_risk_threshold():
    with pytest.raises(ValueError):
        assign_risk_value_segments(
            pd.Series([0.8]),
            pd.Series([100.0]),
            risk_threshold=1.5,
        )


def test_rejects_negative_risk_probability():
    with pytest.raises(ValueError):
        assign_risk_value_segments(
            pd.Series([-0.1]),
            pd.Series([100.0]),
        )


def test_rejects_missing_risk_probability():
    with pytest.raises(ValueError):
        assign_risk_value_segments(
            pd.Series([float("nan")]),
            pd.Series([100.0]),
        )


def test_rejects_missing_clv_threshold_alignment():
    risk = pd.Series([0.8], index=["customer_a"])
    clv = pd.Series([100.0], index=["customer_b"])

    with pytest.raises(ValueError):
        assign_risk_value_segments(risk, clv)
