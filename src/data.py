
import pandas as pd
import pytest

from src.data import (
    load_csv,
    validate_required_columns,
    basic_data_quality_report,
)


def test_validate_required_columns_passes():
    df = pd.DataFrame({
        "customer_id": [1],
        "churn": [0],
    })

    # Should not raise an exception.
    validate_required_columns(
        df, ["customer_id", "churn"]
    )


def test_validate_required_columns_fails():
    df = pd.DataFrame({
        "customer_id": [1],
    })

    with pytest.raises(
        ValueError, match="Missing required columns"
    ):
        validate_required_columns(
            df, ["customer_id", "churn"]
        )


def test_basic_quality_report():
    df = pd.DataFrame({
        "x": [1, 1],
        "y": [2, 2],
    })

    report = basic_data_quality_report(df)

    assert report["row_count"] == 2
    assert report["column_count"] == 2
    assert report["duplicate_rows"] == 1
    assert report["missing_cells"] == 0
    assert report["columns"] == ["x", "y"]


def test_load_csv_reads_data(tmp_path):
    csv_path = tmp_path / "customers.csv"

    pd.DataFrame({
        "customer_id": [1, 2],
        "churn": [0, 1],
    }).to_csv(csv_path, index=False)

    result = load_csv(csv_path)

    assert len(result) == 2
    assert list(result.columns) == [
        "customer_id",
        "churn",
    ]
    assert result["churn"].tolist() == [0, 1]


def test_load_csv_missing_file(tmp_path):
    csv_path = tmp_path / "missing.csv"

    with pytest.raises(
        FileNotFoundError, match="CSV not found"
    ):
        load_csv(csv_path)


def test_load_csv_empty_file(tmp_path):
    csv_path = tmp_path / "empty.csv"

    pd.DataFrame({
        "customer_id": pd.Series(dtype="int64")
    }).to_csv(csv_path, index=False)

    with pytest.raises(
        ValueError, match="CSV contains no rows"
    ):
        load_csv(csv_path)


def test_quality_report_counts_missing_cells():
    df = pd.DataFrame({
        "x": [1, None],
        "y": [None, 2],
    })

    report = basic_data_quality_report(df)

    assert report["row_count"] == 2
    assert report["column_count"] == 2
    assert report["missing_cells"] == 2
    assert report["duplicate_rows"] == 0


def test_quality_report_does_not_modify_input():
    df = pd.DataFrame({
        "x": [1, None],
        "y": [2, 3],
    })

    original = df.copy(deep=True)

    basic_data_quality_report(df)

    pd.testing.assert_frame_equal(df, original)
