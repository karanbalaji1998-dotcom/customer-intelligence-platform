import pandas as pd
import pytest
from src.data import validate_required_columns, basic_data_quality_report

def test_validate_required_columns_passes():
    validate_required_columns(pd.DataFrame({"customer_id": [1], "churn": [0]}), ["customer_id", "churn"])

def test_validate_required_columns_fails():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_required_columns(pd.DataFrame({"customer_id": [1]}), ["customer_id", "churn"])

def test_basic_quality_report():
    df = pd.DataFrame({"x": [1, 1], "y": [2, 2]})
    report = basic_data_quality_report(df)
    assert report["row_count"] == 2
    assert report["duplicate_rows"] == 1
    assert report["missing_cells"] == 0
