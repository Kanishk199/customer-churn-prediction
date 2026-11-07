"""Stratified data splitting and cross-validation generator module."""

import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

from src.config import RANDOM_STATE, TEST_SIZE, N_SPLITS

logger = logging.getLogger(__name__)


def create_stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and test sets using stratified sampling based on the target.

    Stratification ensures that the ~26.5% minority churn class ratio is identical
    in both training and test partitions, preventing sampling bias.

    Args:
        X: Feature DataFrame.
        y: Target Series (0 and 1).
        test_size: Proportion for test partition (default: 0.20).
        random_state: Reproducibility seed.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    train_churn_pct = (y_train.mean()) * 100
    test_churn_pct = (y_test.mean()) * 100

    logger.info(
        "Stratified split completed. Train: %d (%.2f%% churn), Test: %d (%.2f%% churn)",
        len(y_train),
        train_churn_pct,
        len(y_test),
        test_churn_pct,
    )

    # Sanity check: discrepancy should be negligible (< 0.2%)
    if abs(train_churn_pct - test_churn_pct) > 0.5:
        raise ValueError(
            f"Stratification imbalance detected: Train churn={train_churn_pct:.2f}%, Test churn={test_churn_pct:.2f}%"
        )

    return X_train, X_test, y_train, y_test


def get_stratified_kfold(
    n_splits: int = N_SPLITS,
    shuffle: bool = True,
    random_state: int = RANDOM_STATE,
) -> StratifiedKFold:
    """Create a configured StratifiedKFold cross-validator.

    Args:
        n_splits: Number of folds (default: 5).
        shuffle: Whether to shuffle before splitting.
        random_state: Random state seed.

    Returns:
        StratifiedKFold: Ready-to-use cross-validator.
    """
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )


def summarize_split(y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
    """Generate summary statistics for train and test splits."""
    return {
        "train_total": len(y_train),
        "train_churn_count": int(y_train.sum()),
        "train_churn_pct": float(y_train.mean() * 100),
        "test_total": len(y_test),
        "test_churn_count": int(y_test.sum()),
        "test_churn_pct": float(y_test.mean() * 100),
    }
