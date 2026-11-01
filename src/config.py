"""Central configuration and schema definitions.

Update these names after selecting the dataset. Do not put secrets here.
"""
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

@dataclass(frozen=True)
class DataSchema:
    customer_id: str = "customer_id"
    churn_target: str = "churn"
    monthly_revenue: str = "monthly_revenue"
    tenure_months: str = "tenure_months"
    required_columns: tuple[str, ...] = field(
        default=("customer_id", "churn")
    )

SCHEMA = DataSchema()
RANDOM_STATE = 42
TEST_SIZE = 0.20
CLV_HORIZON_MONTHS = 12
