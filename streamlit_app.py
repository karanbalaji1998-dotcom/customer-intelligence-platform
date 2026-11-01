"""Streamlit scaffold for the Customer Intelligence & Retention Platform."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="Customer Intelligence Platform", page_icon="📊", layout="wide")
st.title("Customer Intelligence & Retention Platform")
st.caption("Churn risk • Customer lifetime value • Retention prioritization")

st.info(
    "Starter scaffold: no model has been trained yet. The dashboard will be connected "
    "to validated model artifacts in a later step."
)

left, middle, right = st.columns(3)
left.metric("Customers scored", "—")
middle.metric("High-risk customers", "—")
right.metric("Estimated value at risk", "—")

st.subheader("Planned modules")
st.markdown(
    "- **Churn prediction:** calibrated probabilities and threshold-aware evaluation.\n"
    "- **CLV estimation:** fixed-horizon expected value with explicit assumptions.\n"
    "- **Risk/value segments:** actionable customer prioritization.\n"
    "- **Monitoring:** input validation, data drift checks, and prediction logs."
)
st.warning("Do not interpret placeholder metrics as model results.")
