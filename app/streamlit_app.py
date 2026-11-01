
"""Customer Intelligence & Retention Platform dashboard."""

from pathlib import Path
import sys
import traceback

import joblib
import pandas as pd
import streamlit as st

# Ensure the project root is available for imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring import score_customers

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"

# Dashboard configuration.
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
    [
        "Overview",
        "Customer Risk Scoring",
        "Retention Priorities",
    ],
)

st.sidebar.info(
    "Demo using the IBM Telco Customer Churn dataset. "
    "Financial estimates are illustrative."
)


@st.cache_resource
def load_model():
    """Load the trained churn prediction pipeline."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found. Expected location: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# Initialize session state.
if "scored_customers" not in st.session_state:
    st.session_state["scored_customers"] = None


# -------------------------------------------------------------------
# PAGE 1: OVERVIEW
# -------------------------------------------------------------------

if page == "Overview":
    st.subheader("Platform Overview")

    st.markdown(
        """
        This application demonstrates a customer analytics workflow:

        - Churn prediction using a trained machine-learning pipeline
        - Customer risk and value segmentation
        - Illustrative 12-month revenue and gross-profit estimates
        - Ranked customer retention priorities
        """
    )

    st.warning(
        "Use compatible customer feature columns for scoring. "
        "Validate model performance, probability calibration, and "
        "financial assumptions before real business use."
    )

    scored = st.session_state["scored_customers"]

    if scored is not None and not scored.empty:
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Customers scored",
            f"{len(scored):,}",
        )

        col2.metric(
            "High-risk customers",
            f"{(scored['churn_probability'] >= 0.40).sum():,}",
        )

        col3.metric(
            "Average predicted churn risk",
            f"{scored['churn_probability'].mean():.1%}",
        )

        if "customer_segment" in scored.columns:
            st.subheader("Customer Segments")

            segment_counts = (
                scored["customer_segment"]
                .value_counts()
                .rename_axis("Customer Segment")
                .rename("Customers")
            )

            st.bar_chart(segment_counts)

        st.subheader("Highest-Risk Customers")

        st.dataframe(
            scored.sort_values(
                "churn_probability",
                ascending=False,
            ).head(10),
            use_container_width=True,
        )

    else:
        st.info(
            "No customers scored yet. Open Customer Risk Scoring "
            "to upload a compatible CSV."
        )


# -------------------------------------------------------------------
# PAGE 2: CUSTOMER RISK SCORING
# -------------------------------------------------------------------

elif page == "Customer Risk Scoring":
    st.subheader("Customer Risk Scoring")

    if not MODEL_PATH.is_file():
        st.error(
            "The trained model was not found at "
            "`models/churn_model.joblib`. Ensure the model artifact "
            "is available in the deployment environment."
        )

    else:
        uploaded_file = st.file_uploader(
            "Upload customer feature CSV",
            type=["csv"],
            help=(
                "Upload a CSV with the feature columns expected by "
                "the trained model. customerID and Churn are optional."
            ),
        )

        if uploaded_file is not None:
            try:
                customer_data = pd.read_csv(uploaded_file)

                if customer_data.empty:
                    st.error(
                        "The uploaded CSV contains no rows."
                    )

                else:
                    # Load the trained model and score customers.
                    model = load_model()

                    results = score_customers(
                        model,
                        customer_data,
                    )

                    # Always replace previous results to prevent stale data.
                    st.session_state["scored_customers"] = results

                    # Explain which business outputs are available.
                    if "MonthlyCharges" not in customer_data.columns:
                        st.warning(
                            "MonthlyCharges is missing. Churn-risk "
                            "scores are available, but value-based "
                            "segments and retention priorities "
                            "cannot be calculated."
                        )

                    st.success(
                        f"Successfully scored {len(results):,} customers."
                    )

                    # Summary metrics.
                    metric1, metric2, metric3 = st.columns(3)

                    metric1.metric(
                        "Customers Scored",
                        f"{len(results):,}",
                    )

                    metric2.metric(
                        "High Risk",
                        f"{(results['churn_probability'] >= 0.65).sum():,}",
                    )

                    metric3.metric(
                        "Medium or High Risk",
                        f"{(results['churn_probability'] >= 0.40).sum():,}",
                    )

                    # Results table.
                    st.subheader("Scored Customer Results")

                    st.dataframe(
                        results.sort_values(
                            "churn_probability",
                            ascending=False,
                        ),
                        use_container_width=True,
                    )

                    # Download scores.
                    st.download_button(
                        label="Download Scored Customers",
                        data=results.to_csv(index=False),
                        file_name="customer_churn_scores.csv",
                        mime="text/csv",
                    )

            except Exception as exc:
                st.error(f"Scoring failed: {exc}")

                with st.expander("Technical error details"):
                    st.code(traceback.format_exc())

                st.caption(
                    "Check that the uploaded CSV contains the feature "
                    "columns expected by the trained model, with "
                    "compatible data types."
                )


# -------------------------------------------------------------------
# PAGE 3: RETENTION PRIORITIES
# -------------------------------------------------------------------

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
            "cannot be calculated. Upload a compatible customer CSV."
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

        # Priority summary.
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Customers Ranked",
            f"{len(ranked):,}",
        )

        col2.metric(
            "High-Risk Customers",
            f"{(ranked['churn_probability'] >= 0.40).sum():,}",
        )

        col3.metric(
            "Average Priority Score",
            f"{ranked['retention_priority_score'].mean():,.2f}",
        )

        st.subheader("Ranked Retention Opportunities")

        st.dataframe(
            ranked,
            use_container_width=True,
        )

        st.download_button(
            label="Download Retention Priorities",
            data=ranked.to_csv(index=False),
            file_name="retention_priorities.csv",
            mime="text/csv",
        )

        st.caption(
            "Priority score = predicted churn probability × estimated "
            "12-month gross-profit proxy. This is a prioritization "
            "heuristic, not a causal estimate of campaign savings. "
            "Financial estimates are illustrative and depend on "
            "the validity of the underlying assumptions."
        )
