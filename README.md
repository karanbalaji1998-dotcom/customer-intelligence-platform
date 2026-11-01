# Customer Intelligence & Retention Platform

A GitHub-ready machine learning project for customer churn prediction, customer lifetime value (CLV) estimation, and customer risk/value segmentation.

> **Status:** Starter scaffold only. No model has been trained yet.

## Project goals

- Validate and prepare customer-level data.
- Train and evaluate churn models without data leakage.
- Estimate CLV over an explicitly defined horizon using documented assumptions.
- Combine churn risk and customer value into actionable segments.
- Provide a Streamlit dashboard for exploration and prioritization.
- Add tests, reproducible setup instructions, and model monitoring hooks.

## Repository structure

```text
customer-intelligence-platform/
├── app/
│   └── streamlit_app.py
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
├── notebooks/
│   └── 01_data_understanding.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── churn.py
│   ├── clv.py
│   ├── segmentation.py
│   └── monitoring.py
├── tests/
│   ├── test_data.py
│   ├── test_clv.py
│   └── test_segmentation.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Quick start

Python 3.10+ is recommended.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

### Data

Put a permitted customer-level CSV in `data/raw/` (raw data is ignored by Git). See [`data/README.md`](data/README.md) for the expected fields and privacy guidance.

The starter code does not assume a specific dataset schema. Update `src/config.py` and the feature/data functions after choosing a dataset.

### Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard is a scaffold and will display setup guidance until trained artifacts are available.

### Run tests

```bash
pytest -q
```

## Planned workflow

1. Define the prediction target and observation date.
2. Validate IDs, target labels, missingness, duplicates, and feature types.
3. Split data into train/validation/test sets appropriately; use a time-based split if timestamps are available.
4. Build a preprocessing pipeline fit on training data only.
5. Compare a baseline against a tree-based model.
6. Evaluate discrimination, precision/recall, calibration, and threshold trade-offs.
7. Estimate CLV over a stated horizon with transparent assumptions.
8. Segment customers using churn risk and value; validate segment sizes and business logic.
9. Package the preprocessing and model together, add tests, and document limitations.

## Important modelling notes

- Avoid target leakage: only use features that would be available at scoring time.
- Churn probability is not the same as a causal estimate of retention campaign uplift.
- CLV is an estimate, not observed lifetime revenue. State horizon, margin, discounting, and retention assumptions.
- A high-risk/high-value segment is a prioritization aid, not an automatic instruction to contact a customer.
- Do not commit personal, confidential, or identifiable customer data, credentials, or API keys.

## Current status

This repository contains starter modules and tests only. No model training, performance results, or business impact claims have been produced yet.
