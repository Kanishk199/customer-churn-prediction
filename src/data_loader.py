"""Data loading, cleaning, and feature-target separation module."""

import logging
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_PATH,
    ID_COLUMN,
    TARGET_COLUMN,
    TARGET_MAPPING,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)

logger = logging.getLogger(__name__)


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw dataset from CSV without modifying the source file.

    Args:
        filepath: Optional custom Path to CSV. Defaults to RAW_DATA_PATH.

    Returns:
        pd.DataFrame: Loaded raw dataset.

    Raises:
        FileNotFoundError: If the specified CSV does not exist.
        ValueError: If the dataset does not contain expected columns.
    """
    path = Path(filepath) if filepath else RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    # Read CSV without altering source
    df = pd.read_csv(path)
    logger.info("Loaded raw dataset from %s (shape: %s)", path, df.shape)

    # Validate essential columns
    required_cols = [ID_COLUMN, TARGET_COLUMN] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data types and handle missing/whitespace values.

    Specific treatments:
    - TotalCharges is stored as object with whitespace strings ' ' for new users (tenure == 0).
    - Convert TotalCharges to float64 and impute missing values with 0.0.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame copy with validated types.
    """
    cleaned_df = df.copy()

    # Convert TotalCharges to numeric, coercing invalid/whitespace strings to NaN
    cleaned_df["TotalCharges"] = pd.to_numeric(cleaned_df["TotalCharges"], errors="coerce")

    # Impute missing TotalCharges with 0.0 (empirically corresponding to tenure == 0)
    missing_total_charges = cleaned_df["TotalCharges"].isnull().sum()
    if missing_total_charges > 0:
        cleaned_df["TotalCharges"] = cleaned_df["TotalCharges"].fillna(0.0)
        logger.info(
            "Imputed %d missing/whitespace TotalCharges values with 0.0.",
            missing_total_charges,
        )

    # Ensure SeniorCitizen is treated consistently (int or category)
    cleaned_df["SeniorCitizen"] = cleaned_df["SeniorCitizen"].astype(int)

    return cleaned_df


def separate_features_target(
    df: pd.DataFrame,
    id_col: str = ID_COLUMN,
    target_col: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Remove customer identifier and separate predictive features from target variable.

    Args:
        df: Input cleaned DataFrame.
        id_col: Name of the identifier column to drop.
        target_col: Name of the target column.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X_features, y_target_encoded)

    Raises:
        KeyError: If target_col is missing from DataFrame.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    # Target extraction & binary encoding (Yes -> 1, No -> 0)
    target_series = df[target_col].copy()
    if pd.api.types.is_numeric_dtype(target_series):
        y = target_series.astype(int)
    else:
        y = target_series.map(TARGET_MAPPING)
        if y.isnull().any():
            unmapped = target_series[y.isnull()].unique()
            raise ValueError(f"Unrecognized target values in '{target_col}': {unmapped}")
        y = y.astype(int)

    # Drop target and customerID to prevent data leakage and memorization
    cols_to_drop = [target_col]
    if id_col in df.columns:
        cols_to_drop.append(id_col)

    X = df.drop(columns=cols_to_drop).copy()
    logger.info(
        "Separated features and target: X shape=%s, y shape=%s. Dropped: %s",
        X.shape,
        y.shape,
        cols_to_drop,
    )

    return X, y


def load_and_prepare_data(filepath: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """End-to-end convenience loader: loads, cleans, and separates X and y.

    Args:
        filepath: Optional path to raw dataset.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y)
    """
    raw_df = load_raw_data(filepath)
    clean_df = clean_dataset(raw_df)
    return separate_features_target(clean_df)
