
"""Customer Intelligence & Retention Platform dashboard."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Intelligence & Retention Platform")
st.caption("Churn risk • Customer value • Retention prioritization")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Overview", "Customer Risk Scoring", "Retention Priorities"],
)

st.sidebar.info(
    "Demo using the IBM Telco Customer Churn dataset. "
    "Financial estimates are illustrative."
)



@st.cache_resource
def load_model():
    # model_path = PROJECT_ROOT / "models" / "churn_model.joblib"

    if not MODEL_PATH.is_file():
        st.error(f"Model file not found. Expected location: {model_path}")
        st.write("Project root:", str(PROJECT_ROOT))
        st.write("Models folder exists:", (PROJECT_ROOT / "models").is_dir())
        st.stop()

    return joblib.load(MODEL_PATH)



def assign_segments(data, risk_threshold=0.40):
    """Create transparent risk/value segments."""
    high_value_threshold = data["MonthlyCharges"].quantile(0.75)

    high_risk = data["churn_probability"] >= risk_threshold
    high_value = data["MonthlyCharges"] >= high_value_threshold

    data = data.copy()
    data["customer_segment"] = np.select(
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
    return data


if "scored_customers" not in st.session_state:
    st.session_state["scored_customers"] = None


if page == "Overview":
    st.subheader("Platform Overview")
    st.markdown(
        """
        This application demonstrates a customer analytics workflow:

        - Churn prediction using a trained machine-learning pipeline
        - Customer risk/value segmentation
        - Illustrative 12-month revenue and gross-profit estimates
        - Ranked customer retention priorities
        """
    )

    st.warning(
        "Use compatible customer feature columns for scoring. "
        "Predicted probabilities and financial estimates require validation "
        "before real business use."
    )

    scored = st.session_state["scored_customers"]

    if scored is not None and not scored.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Customers scored", f"{len(scored):,}")
        c2.metric(
            "High-risk customers",
            f"{(scored['churn_probability'] >= 0.40).sum():,}",
        )
        c3.metric(
            "Average predicted churn risk",
            f"{scored['churn_probability'].mean():.1%}",
        )

        st.subheader("Customer segments")
        st.bar_chart(
            scored["customer_segment"].value_counts()
        )
    else:
        st.info(
            "No customers scored yet. Open Customer Risk Scoring "
            "to upload a compatible CSV."
        )


elif page == "Customer Risk Scoring":
    st.subheader("Customer Risk Scoring")

    if not MODEL_PATH.exists():
        st.error(
            "Model file not found at models/churn_model.joblib. "
            "The trained model must be made available before scoring."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload customer feature CSV",
            type=["csv"],
        )

        if uploaded_file is not None:
            try:
                customer_data = pd.read_csv(uploaded_file)

                if customer_data.empty:
                    st.error("The uploaded CSV contains no rows.")
                    st.stop()

                customer_ids = (
                    customer_data["customerID"].astype(str)
                    if "customerID" in customer_data.columns
                    else pd.Series(
                        [f"Customer_{i+1}" for i in range(len(customer_data))]
                    )
                )

                features = customer_data.drop(
                    columns=["customerID", "Churn"],
                    errors="ignore",
                )

                model = load_model()
                probabilities = model.predict_proba(features)[:, 1]

                results = pd.DataFrame(
                    {
                        "customerID": customer_ids.to_numpy(),
                        "churn_probability": probabilities,
                    }
                )

                results["risk_band"] = np.select(
                    [
                        results["churn_probability"] >= 0.65,
                        results["churn_probability"] >= 0.40,
                    ],
                    ["High", "Medium"],
                    default="Low",
                )

                # MonthlyCharges is needed for value segmentation.
                if "MonthlyCharges" in features.columns:
                    results["MonthlyCharges"] = features[
                        "MonthlyCharges"
                    ].to_numpy()
                    results = assign_segments(results)

                    results["expected_active_months_12m"] = (
                        12 * (
                            1 - results["churn_probability"]
                        ) ** (1 / 12)
                    )
                    results["expected_revenue_12m"] = (
                        results["MonthlyCharges"]
                        * results["expected_active_months_12m"]
                    )
                    results["estimated_gross_profit_12m"] = (
                        results["expected_revenue_12m"] * 0.40
                    )
                    results["retention_priority_score"] = (
                        results["churn_probability"]
                        * results["estimated_gross_profit_12m"]
                    )

                st.session_state["scored_customers"] = results

                st.success(f"Successfully scored {len(results):,} customers.")

                st.dataframe(
                    results.sort_values(
                        "churn_probability", ascending=False
                    ),
                    use_container_width=True,
                )

                st.download_button(
                    "Download scored customers",
                    data=results.to_csv(index=False),
                    file_name="customer_churn_scores.csv",
                    mime="text/csv",
                )

            except Exception as exc:
                st.error(f"Scoring failed: {exc}")
                import traceback
                st.error("Scoring failed. Full error details:")
                st.code(traceback.format_exc())
                st.caption(
                    "Verify the feature names, column types, and required "
                    "input columns match the training pipeline."
                )


elif page == "Retention Priorities":
    st.subheader("Retention Priorities")

    scored = st.session_state["scored_customers"]

    if scored is None or scored.empty:
        st.info(
            "Score a customer CSV first. Results will appear here "
            "for the current dashboard session."
        )
    elif "retention_priority_score" not in scored.columns:
        st.warning(
            "MonthlyCharges was missing, so value-based prioritization "
            "cannot be calculated. Upload compatible features."
        )
    else:
        ranked = scored.sort_values(
            "retention_priority_score",
            ascending=False,
        ).copy()
        ranked.insert(
            0,
            "priority_rank",
            range(1, len(ranked) + 1),
        )

        st.dataframe(ranked, use_container_width=True)

        st.download_button(
            "Download retention priorities",
            data=ranked.to_csv(index=False),
            file_name="retention_priorities.csv",
            mime="text/csv",
        )

        st.caption(
            "Priority score = predicted churn probability × estimated "
            "12-month gross-profit proxy. This is a heuristic, not "
            "estimated campaign savings."
        )
