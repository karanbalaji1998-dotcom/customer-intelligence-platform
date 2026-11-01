
"""Customer scoring and retention-priority calculations."""

import numpy as np
import pandas as pd


def score_customers(model, customer_data):
    """Score customer records and calculate illustrative retention metrics.

    customer_data should contain the features expected by the trained
    model. An optional customerID column is retained as an identifier;
    customerID and Churn are excluded from model features.

    The financial estimates assume the model probability represents
    12-month churn risk. Validate this assumption before business use.
    """
    if not isinstance(customer_data, pd.DataFrame):
        raise TypeError("customer_data must be a pandas DataFrame")

    if customer_data.empty:
        raise ValueError("customer_data contains no rows")

    if customer_data.columns.duplicated().any():
        raise ValueError("customer_data contains duplicate columns")

    customer_ids = (
        customer_data["customerID"].astype(str).to_numpy()
        if "customerID" in customer_data.columns
        else np.array(
            [f"Customer_{i + 1}" for i in range(len(customer_data))]
        )
    )

    features = customer_data.drop(
        columns=["customerID", "Churn"],
        errors="ignore",
    )

    if features.empty:
        raise ValueError("No model features were provided")

    if features.isna().all(axis=0).any():
        empty_columns = features.columns[
            features.isna().all(axis=0)
        ].tolist()
        raise ValueError(
            f"Feature columns contain only missing values: {empty_columns}"
        )

    if not hasattr(model, "predict_proba"):
        raise TypeError("Model must support predict_proba")

    probabilities = np.asarray(model.predict_proba(features))

    if probabilities.ndim != 2 or probabilities.shape[0] != len(features):
        raise ValueError("Model returned an invalid probability shape")

    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        final_step = list(model.named_steps.values())[-1]
        classes = getattr(final_step, "classes_", None)

    if classes is None:
        raise ValueError("Cannot identify the model's class labels")

    churn_columns = np.flatnonzero(np.asarray(classes) == 1)
    if len(churn_columns) != 1:
        raise ValueError("Model must have exactly one class labelled 1 for churn")

    if probabilities.shape[1] != len(classes):
        raise ValueError("Probability columns do not match model classes")

    churn_probability = probabilities[:, churn_columns[0]]

    if (
        not np.isfinite(churn_probability).all()
        or (churn_probability < 0).any()
        or (churn_probability > 1).any()
    ):
        raise ValueError("Model returned invalid churn probabilities")

    results = pd.DataFrame({
        "customerID": customer_ids,
        "churn_probability": churn_probability,
    })

    results["risk_band"] = np.select(
        [
            results["churn_probability"] >= 0.65,
            results["churn_probability"] >= 0.40,
        ],
        ["High", "Medium"],
        default="Low",
    )

    if "MonthlyCharges" in features.columns:
        monthly_charges = pd.to_numeric(
            features["MonthlyCharges"], errors="coerce"
        ).to_numpy(dtype=float)

        if (
            not np.isfinite(monthly_charges).all()
            or (monthly_charges < 0).any()
        ):
            raise ValueError(
                "MonthlyCharges must contain finite, non-negative values"
            )

        results["MonthlyCharges"] = monthly_charges

        # Keep segment thresholds and labels consistent with the dashboard.
        high_value_threshold = np.quantile(monthly_charges, 0.75)
        high_risk = results["churn_probability"] >= 0.40
        high_value = results["MonthlyCharges"] >= high_value_threshold

        results["customer_segment"] = np.select(
            [
                high_risk & high_value,
                high_risk & ~high_value,
                ~high_risk & high_value,
            ],
            [
                "High Risk - High Value",
                "High Risk - Lower Value",
                "Low Risk - High Value",
            ],
            default="Low Risk - Lower Value",
        )

        p = results["churn_probability"].to_numpy()
        monthly_survival = (1 - p) ** (1 / 12)

        expected_months = np.zeros_like(p, dtype=float)
        valid = p < 1
        expected_months[valid] = (
            p[valid] / (1 - monthly_survival[valid])
        )

        results["expected_active_months_12m"] = expected_months
        results["expected_revenue_12m"] = (
            monthly_charges * expected_months
        )
        results["estimated_gross_profit_12m"] = (
            results["expected_revenue_12m"] * 0.40
        )
        results["retention_priority_score"] = (
            p * results["estimated_gross_profit_12m"]
        )

    return results
