# Customer Intelligence & Retention Platform

An end-to-end machine learning project for **customer churn prediction, risk/value segmentation, and retention prioritization**, built using Python, scikit-learn, pandas, and Streamlit.

**Live dashboard:** [Open the Customer Intelligence & Retention Platform](https://customer-intelligence-platform-fqwitkahhzqqkd8hts9yy5.streamlit.app/)

**Repository:** [GitHub source code](https://github.com/karanbalaji1998-dotcom/customer-intelligence-platform)

## Project Overview

Customer churn can reduce recurring revenue and make customer acquisition investments less effective. This project demonstrates how machine learning can help identify customers at elevated churn risk and prioritize them for further retention analysis.

The platform combines a trained classification pipeline with customer segmentation and illustrative financial estimates in an interactive dashboard.

## Key Features

- **Data preparation:** Handles missing values, numeric conversion, and mixed numerical/categorical customer features.
- **Churn prediction:** Uses a scikit-learn pipeline with preprocessing and a Random Forest classifier.
- **Model comparison:** Evaluates Logistic Regression and Random Forest using classification and ranking metrics.
- **Risk scoring:** Generates customer-level churn probabilities.
- **Risk/value segmentation:** Groups customers using predicted churn risk and monthly charges.
- **Retention prioritization:** Ranks customers using a score combining predicted churn probability and estimated gross profit.
- **Interactive dashboard:** Provides customer scoring and retention-priority exploration through Streamlit.

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- Streamlit
- Joblib
- Matplotlib
- GitHub
- Streamlit Community Cloud

## Model Development and Evaluation

The project uses the IBM Telco Customer Churn dataset, a public dataset containing customer attributes, service subscriptions, billing information, and churn labels.

The modelling workflow includes:

1. Data inspection and preparation.
2. Stratified train/test splitting.
3. Numerical imputation and scaling.
4. Categorical imputation and one-hot encoding.
5. Logistic Regression baseline.
6. Random Forest classification.
7. Evaluation using precision, recall, F1-score, ROC-AUC, and PR-AUC.
8. Threshold exploration to understand the trade-off between identifying churners and generating false alarms.

The preprocessing steps are packaged with the classifier in a scikit-learn pipeline to ensure consistent transformations during scoring.

### Experimental Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.806 | 0.772 |
| Churn precision | 0.657 | 0.553 |
| Churn recall | 0.559 | 0.735 |
| Churn F1-score | 0.604 | 0.631 |
| ROC-AUC | 0.842 | 0.842 |
| PR-AUC | 0.633 | 0.648 |

In this experiment, Random Forest achieved higher churn recall and F1-score, while Logistic Regression achieved higher accuracy and churn precision. Random Forest may be preferable when identifying more potentially churning customers is the priority, subject to the cost of false positives.

These are experimental results on a held-out split, not evidence of performance in a live production environment. Threshold exploration and any subsequent model selection should be evaluated on separate validation data to avoid optimistic estimates.

## Customer Segmentation

Customers are categorized using predicted churn risk and monthly charges:

- **High Risk – High Value:** Customers with elevated churn risk and higher monthly charges.
- **High Risk – Lower Value:** Customers with elevated churn risk and lower monthly charges.
- **Low Risk – High Value:** Customers with lower predicted churn risk and higher monthly charges.
- **Low Risk – Lower Value:** Customers with lower predicted churn risk and lower monthly charges.

These segments support prioritization and exploration. Monthly charges are used as a simple value proxy rather than a comprehensive measure of customer profitability.

## Retention Prioritization

The current retention priority score is:

`Retention Priority Score = Predicted Churn Probability × Estimated Gross Profit`

Estimated gross profit uses monthly charges, estimated active months, and an assumed 40% gross margin over a 12-month horizon.

**Important assumptions and limitations:**

- The classifier predicts the dataset's churn label; its probability is not automatically a calibrated 12-month churn probability.
- Expected active months and financial estimates are illustrative until the time horizon and probability calibration are validated.
- Monthly charges are not the same as realized revenue or customer lifetime value.
- The priority score is a ranking heuristic, not a causal estimate of how much revenue a retention campaign will save.
- No retention uplift or realized financial impact has been established.

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/karanbalaji1998-dotcom/customer-intelligence-platform.git
cd customer-intelligence-platform
```

### 2. Install dependencies

Python 3.12 is recommended to match the deployed application environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

### 3. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

Ensure the trained model artifact is present at `models/churn_model.joblib`.

### 4. Score customers

Upload a CSV containing the model's expected input features. The file should include the required customer attributes and must not include the target label as an input feature.

## Repository Structure

```text
customer-intelligence-platform/
├── app/
│   └── streamlit_app.py
├── data/
│   └── README.md
├── models/
│   └── churn_model.joblib
├── notebooks/
│   └── 01_data_understanding.ipynb
├── src/
├── tests/
├── requirements.txt
└── README.md
```

The structure above summarizes the principal project components. See the repository for the complete file listing.

## Future Improvements

- Calibrate churn probabilities and define a clear prediction horizon.
- Tune classification thresholds on validation data.
- Add calibration plots and precision-recall curves.
- Add automated tests for data validation, segmentation, and financial calculations.
- Add model and data drift monitoring.
- Incorporate intervention costs and experimentally measured retention uplift.
- Develop a more defensible customer lifetime value model using customer-level revenue and retention history.

## Disclaimer

This project is a portfolio demonstration using a public dataset. Financial estimates are illustrative and should not be treated as validated customer lifetime value, realized business outcomes, or production retention recommendations.

No confidential customer data or credentials should be committed to the repository.
