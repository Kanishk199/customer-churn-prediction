"""Configuration settings and constants for the customer churn prediction pipeline."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "Telco-Customer-Churn.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Target & Identifier
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
TARGET_MAPPING = {"Yes": 1, "No": 0}

# Feature Classification
NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Splitting & Cross-Validation Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
