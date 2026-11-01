
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
        "Model Evaluation",
        "Business Impact Simulator"
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
    
# -------------------------------------------------------------------
# PAGE 4: MODEL EVALUATION
# -------------------------------------------------------------------

elif page == "Model Evaluation":
    st.subheader("Model Evaluation")
    st.caption("Offline evaluation results from the held-out test dataset.")

    # Load the saved model comparison table.
    METRICS_PATH = PROJECT_ROOT / "model_comparison.csv"

    if not METRICS_PATH.is_file():
        st.error(
            "Model evaluation results were not found. "
            "Ensure model_comparison.csv is committed to the "
            "repository root."
        )

    else:
        try:
            comparison = pd.read_csv(
                METRICS_PATH,
                index_col=0,
            )

            if comparison.empty:
                st.warning("The model comparison file contains no results.")

            else:
                # Display the complete model comparison.
                st.subheader("Model Comparison")
                st.dataframe(
                    comparison.style.format("{:.3f}"),
                    use_container_width=True,
                )

                # Let the user select a model for detailed metrics.
                selected_model = st.selectbox(
                    "Select model",
                    comparison.index.tolist(),
                )

                metrics = comparison.loc[selected_model]

                st.subheader(f"Performance: {selected_model}")

                preferred_metrics = [
                    "Accuracy",
                    "Churn Precision",
                    "Churn Recall",
                    "Churn F1",
                    "ROC-AUC",
                    "PR-AUC (Average Precision)",
                ]

                available_metrics = [
                    metric
                    for metric in preferred_metrics
                    if metric in metrics.index
                ]

                metric_columns = st.columns(
                    min(len(available_metrics), 3)
                )

                for i, metric in enumerate(available_metrics):
                    with metric_columns[i % len(metric_columns)]:
                        st.metric(
                            label=metric,
                            value=f"{metrics[metric]:.3f}",
                        )

                # Compare discrimination metrics visually.
                chart_metrics = [
                    metric
                    for metric in [
                        "ROC-AUC",
                        "PR-AUC (Average Precision)",
                    ]
                    if metric in comparison.columns
                ]

                if chart_metrics:
                    st.subheader("Model Discrimination")

                    st.bar_chart(
                        comparison[chart_metrics]
                    )

                st.info(
                    "These are offline test-set metrics, not live "
                    "production performance. Precision and recall "
                    "depend on the prediction threshold. ROC-AUC and "
                    "PR-AUC use predicted probabilities."
                )

        except (ValueError, KeyError, OSError) as exc:
            st.error(
                f"Unable to read model evaluation results: {exc}"
            )


# -------------------------------------------------------------------
# PAGE 5: BUSINESS IMPACT SIMULATOR
# -------------------------------------------------------------------

elif page == "Business Impact Simulator":
    st.subheader("Business Impact Simulator")

    st.markdown(
        """
        Explore hypothetical customer-retention campaign scenarios.
        Adjust the budget, contact cost, and campaign success rate to
        estimate the potential financial impact.
        """
    )

    scored = st.session_state.get("scored_customers")

    if scored is None or scored.empty:
        st.info(
            "First upload a customer CSV on the Customer Risk Scoring "
            "page to generate customer-level estimates."
        )

    elif "estimated_gross_profit_12m" not in scored.columns:
        st.warning(
            "Financial estimates are unavailable. Upload a compatible "
            "CSV containing MonthlyCharges and score it again."
        )

    else:
        # Scenario assumptions.
        st.sidebar.subheader("Campaign Assumptions")

        budget = st.sidebar.number_input(
            "Campaign budget (₹)",
            min_value=0.0,
            value=100000.0,
            step=10000.0,
        )

        contact_cost = st.sidebar.number_input(
            "Cost per contacted customer (₹)",
            min_value=0.0,
            value=100.0,
            step=10.0,
        )

        success_rate = st.sidebar.slider(
            "Assumed campaign success rate (%)",
            min_value=0,
            max_value=100,
            value=15,
            step=5,
        ) / 100

        risk_threshold = st.slider(
            "Minimum churn probability for targeting",
            min_value=0.0,
            max_value=1.0,
            value=0.40,
            step=0.05,
        )

        # Target only customers above the selected risk threshold.
        targets = scored.loc[
            scored["churn_probability"] >= risk_threshold
        ].copy()

        # Limit campaign size by the available budget.
        if contact_cost > 0:
            affordable_contacts = int(budget // contact_cost)
            campaign = targets.sort_values(
                "retention_priority_score",
                ascending=False,
            ).head(affordable_contacts)
        else:
            campaign = targets.copy() if budget > 0 else targets.iloc[0:0].copy()

        contacted = len(campaign)
        campaign_cost = contacted * contact_cost

        # These are scenario estimates, not causal model predictions.
        assumed_retained_customers = contacted * success_rate

        average_profit_at_risk = (
            campaign["estimated_gross_profit_12m"].mean()
            if contacted > 0
            else 0.0
        )

        potential_profit_retained = (
            assumed_retained_customers * average_profit_at_risk
        )

        estimated_net_impact = potential_profit_retained - campaign_cost

        # Display results.
        st.subheader("Scenario Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Eligible High-Risk Customers",
            f"{len(targets):,}",
        )

        col2.metric(
            "Customers Contacted",
            f"{contacted:,}",
        )

        col3.metric(
            "Estimated Campaign Cost",
            f"₹{campaign_cost:,.0f}",
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Assumed Retained Customers",
            f"{assumed_retained_customers:,.1f}",
        )

        col5.metric(
            "Potential Profit Retained",
            f"₹{potential_profit_retained:,.0f}",
        )

        col6.metric(
            "Estimated Net Impact",
            f"₹{estimated_net_impact:,.0f}",
        )

        # Visualize the scenario.
        chart_data = pd.DataFrame(
            {
                "Amount (₹)": [
                    campaign_cost,
                    potential_profit_retained,
                    estimated_net_impact,
                ]
            },
            index=[
                "Campaign Cost",
                "Potential Profit Retained",
                "Estimated Net Impact",
            ],
        )

        st.subheader("Financial Scenario Breakdown")
        st.bar_chart(chart_data)

        st.subheader("Proposed Campaign Audience")

        if contacted > 0:
            st.dataframe(
                campaign[
                    [
                        "customerID",
                        "churn_probability",
                        "risk_band",
                        "estimated_gross_profit_12m",
                        "retention_priority_score",
                    ]
                ],
                use_container_width=True,
            )

            st.download_button(
                "Download Proposed Campaign Audience",
                data=campaign.to_csv(index=False),
                file_name="proposed_retention_campaign.csv",
                mime="text/csv",
            )
        else:
            st.info(
                "No customers selected. Increase the budget or "
                "lower the risk threshold."
            )

        st.warning(
            "Illustrative scenario only: the success rate is a user "
            "assumption, not a measured campaign uplift. The calculation "
            "assumes estimated gross profit represents profit at risk "
            "and that a successful intervention preserves it. Actual "
            "incremental impact requires a controlled campaign experiment. "
            "Model probabilities may not be calibrated."
        )




