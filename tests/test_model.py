
"""Tests for churn model prediction behavior using synthetic data."""

import numpy as np
import pandas as pd
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@pytest.fixture
def synthetic_churn_model():
    """Build a small churn model without real customer data."""

    data = pd.DataFrame({
        "tenure": [1, 2, 3, 5, 12, 18, 24, 36, 48, 60, 65, 70],
        "MonthlyCharges": [
            90, 85, 95, 80, 70, 65, 60, 55, 50, 45, 40, 35
        ],
        "Contract": [
            "Month-to-month", "Month-to-month", "Month-to-month",
            "Month-to-month", "One year", "One year",
            "Month-to-month", "One year", "Two year",
            "Two year", "Two year", "Two year"
        ],
    })

    target = np.array([
        1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
    ])

    numeric_features = ["tenure", "MonthlyCharges"]
    categorical_features = ["Contract"]

    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(data, target)

    return model, data


def test_model_predicts_binary_classes(synthetic_churn_model):
    model, data = synthetic_churn_model

    predictions = model.predict(data)

    assert len(predictions) == len(data)
    assert set(predictions).issubset({0, 1})


def test_model_returns_valid_probabilities(synthetic_churn_model):
    model, data = synthetic_churn_model

    probabilities = model.predict_proba(data)

    assert probabilities.shape == (len(data), 2)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(len(data)),
    )


def test_model_handles_unseen_categories(synthetic_churn_model):
    model, data = synthetic_churn_model

    new_customer = data.iloc[[0]].copy()
    new_customer["Contract"] = "New contract type"

    probabilities = model.predict_proba(new_customer)

    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()


def test_model_repeated_predictions_are_consistent(
    synthetic_churn_model,
):
    model, data = synthetic_churn_model

    first = model.predict_proba(data)
    second = model.predict_proba(data)

    np.testing.assert_allclose(first, second)
