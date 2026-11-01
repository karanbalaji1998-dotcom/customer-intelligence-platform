
"""Tests for customer scoring using synthetic data."""

import numpy as np
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from src.scoring import score_customers


@pytest.fixture
def synthetic_model():
    """Train a small test-only classifier using artificial customers."""
    X = pd.DataFrame({
        "tenure": [1, 2, 3, 6, 12, 18, 24, 36, 48, 60, 65, 70],
        "MonthlyCharges": [
            90, 85, 95, 80, 70, 65, 60, 55, 50, 45, 40, 35
        ],
    })
    y = np.array([1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0])

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X, y)
    return model


@pytest.fixture
def synthetic_customers():
    return pd.DataFrame({
        "customerID": ["SYN001", "SYN002", "SYN003"],
        "tenure": [2, 24, 60],
        "MonthlyCharges": [90.0, 60.0, 40.0],
    })


def test_scores_synthetic_customers(synthetic_model, synthetic_customers):
    results = score_customers(synthetic_model, synthetic_customers)

    assert len(results) == 3
    assert results["customerID"].tolist() == [
        "SYN001", "SYN002", "SYN003"
    ]
    assert results["churn_probability"].between(0, 1).all()


def test_outputs_expected_risk_bands(synthetic_model, synthetic_customers):
    results = score_customers(synthetic_model, synthetic_customers)

    assert set(results["risk_band"]).issubset({"Low", "Medium", "High"})


def test_outputs_retention_metrics(synthetic_model, synthetic_customers):
    results = score_customers(synthetic_model, synthetic_customers)

    expected_columns = [
        "expected_active_months_12m",
        "expected_revenue_12m",
        "estimated_gross_profit_12m",
        "retention_priority_score",
        "customer_segment",
    ]

    for column in expected_columns:
        assert column in results.columns
        assert np.isfinite(results[column]).all() if column != "customer_segment" else results[column].notna().all()

    for column in [
        "expected_active_months_12m",
        "expected_revenue_12m",
        "estimated_gross_profit_12m",
        "retention_priority_score",
    ]:
        assert (results[column] >= 0).all()


def test_generates_ids_when_customer_id_missing(synthetic_model):
    customers = pd.DataFrame({
        "tenure": [2, 24],
        "MonthlyCharges": [90.0, 60.0],
    })

    results = score_customers(synthetic_model, customers)

    assert results["customerID"].tolist() == [
        "Customer_1", "Customer_2"
    ]


def test_rejects_empty_input(synthetic_model):
    customers = pd.DataFrame(
        columns=["tenure", "MonthlyCharges"]
    )

    with pytest.raises(ValueError, match="no rows"):
        score_customers(synthetic_model, customers)


def test_rejects_missing_feature_values(synthetic_model):
    customers = pd.DataFrame({
        "tenure": [None],
        "MonthlyCharges": [90.0],
    })

    with pytest.raises(ValueError, match="only missing values"):
        score_customers(synthetic_model, customers)


def test_rejects_negative_monthly_charges(synthetic_model):
    customers = pd.DataFrame({
        "tenure": [12],
        "MonthlyCharges": [-10.0],
    })

    with pytest.raises(ValueError, match="MonthlyCharges"):
        score_customers(synthetic_model, customers)


def test_does_not_modify_input(synthetic_model, synthetic_customers):
    original = synthetic_customers.copy(deep=True)

    score_customers(synthetic_model, synthetic_customers)

    pd.testing.assert_frame_equal(synthetic_customers, original)
