"""High-level pipeline assembly, data transformation orchestration, and validation checks."""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.config import RANDOM_STATE, TEST_SIZE, N_SPLITS
from src.data_loader import load_and_prepare_data
from src.preprocessor import (
    build_preprocessor,
    get_feature_names,
    transform_to_dataframe,
)
from src.split import create_stratified_split, get_stratified_kfold, summarize_split

logger = logging.getLogger(__name__)


def create_feature_pipeline() -> Pipeline:
    """Create a baseline pipeline containing the complete feature preprocessor.

    Estimators can subsequently be appended in Phase 3 (e.g. pipeline.steps.append(('clf', model))).

    Returns:
        Pipeline: Scikit-learn Pipeline with 'preprocessor' step.
    """
    preprocessor = build_preprocessor()
    return Pipeline(steps=[("preprocessor", preprocessor)])


def fit_and_transform_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    preprocessor: ColumnTransformer = None,
) -> Tuple[ColumnTransformer, pd.DataFrame, pd.DataFrame]:
    """Fit preprocessor strictly on X_train, then transform both X_train and X_test.

    Prevents data leakage by ensuring all imputer statistics and scaling parameters
    (mean, variance) are learned solely from the training distribution.

    Args:
        X_train: Raw training features.
        X_test: Raw testing features.
        preprocessor: Optional ColumnTransformer instance.

    Returns:
        Tuple[ColumnTransformer, pd.DataFrame, pd.DataFrame]:
            (fitted_preprocessor, X_train_transformed_df, X_test_transformed_df)
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()

    # CRITICAL: Fit ONLY on training partition
    preprocessor.fit(X_train)

    # Transform both partitions using the trained parameters
    X_train_trans = transform_to_dataframe(preprocessor, X_train)
    X_test_trans = transform_to_dataframe(preprocessor, X_test)

    return preprocessor, X_train_trans, X_test_trans


def run_pipeline_validation() -> Dict[str, Any]:
    """Execute rigorous diagnostic checks on data loading, preprocessing, and CV readiness.

    Returns:
        Dict[str, Any]: Detailed diagnostic summary report.
    """
    results: Dict[str, Any] = {}

    # 1. Load and prepare raw data
    X, y = load_and_prepare_data()
    results["dataset_shape"] = {"samples": X.shape[0], "features": X.shape[1]}
    results["customerID_removed"] = "customerID" not in X.columns
    results["target_distribution"] = {
        "churn_1_count": int((y == 1).sum()),
        "churn_0_count": int((y == 0).sum()),
        "churn_rate_pct": float(y.mean() * 100),
    }

    # 2. Stratified train/test split
    X_train, X_test, y_train, y_test = create_stratified_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    results["split_summary"] = summarize_split(y_train, y_test)

    # 3. Fit strictly on train and transform both
    preprocessor, X_train_trans, X_test_trans = fit_and_transform_split(X_train, X_test)
    feature_names = get_feature_names(preprocessor)

    results["transformed_features_count"] = len(feature_names)
    results["feature_names"] = feature_names
    results["train_transformed_shape"] = X_train_trans.shape
    results["test_transformed_shape"] = X_test_trans.shape

    # 4. Dimension & Feature Compatibility Check
    compatible_columns = list(X_train_trans.columns) == list(X_test_trans.columns)
    results["columns_match"] = compatible_columns

    # 5. Missing / Infinite values check
    train_nulls = int(X_train_trans.isnull().sum().sum())
    test_nulls = int(X_test_trans.isnull().sum().sum())
    train_infs = int(np.isinf(X_train_trans.values).sum())
    test_infs = int(np.isinf(X_test_trans.values).sum())
    results["null_values"] = {"train_nulls": train_nulls, "test_nulls": test_nulls}
    results["infinite_values"] = {"train_infs": train_infs, "test_infs": test_infs}

    # 6. StandardScaler Verification (Train mean ~ 0, std ~ 1)
    train_means = X_train_trans[["tenure", "MonthlyCharges", "TotalCharges"]].mean().to_dict()
    train_stds = X_train_trans[["tenure", "MonthlyCharges", "TotalCharges"]].std().to_dict()
    results["scaling_check"] = {"means": train_means, "stds": train_stds}

    # 7. Unseen category resilience test
    mock_test = X_test.copy().head(5)
    mock_test["Contract"] = "Decade-to-decade"  # completely unseen category
    mock_trans = transform_to_dataframe(preprocessor, mock_test)
    results["unseen_category_handled"] = (
        mock_trans.shape[1] == len(feature_names) and not mock_trans.isnull().any().any()
    )

    # 8. Stratified 5-Fold Cross-Validation Preparation Check
    cv = get_stratified_kfold(n_splits=N_SPLITS, random_state=RANDOM_STATE)
    cv_fold_shapes = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        fold_X_tr, fold_y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        fold_X_val, fold_y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        fold_preprocessor = build_preprocessor()
        fold_preprocessor.fit(fold_X_tr)

        fold_tr_trans = fold_preprocessor.transform(fold_X_tr)
        fold_val_trans = fold_preprocessor.transform(fold_X_val)

        cv_fold_shapes.append(
            {
                "fold": fold,
                "train_samples": fold_tr_trans.shape[0],
                "val_samples": fold_val_trans.shape[0],
                "features": fold_tr_trans.shape[1],
                "val_churn_pct": float(fold_y_val.mean() * 100),
            }
        )

    results["cv_verification"] = cv_fold_shapes
    results["all_checks_passed"] = (
        results["customerID_removed"]
        and results["columns_match"]
        and train_nulls == 0
        and test_nulls == 0
        and train_infs == 0
        and test_infs == 0
        and results["unseen_category_handled"]
        and len(cv_fold_shapes) == N_SPLITS
    )

    return results


if __name__ == "__main__":
    report = run_pipeline_validation()
    print("=== PIPELINE VALIDATION RESULTS ===")
    print(f"Dataset Shape: {report['dataset_shape']}")
    print(f"customerID removed: {report['customerID_removed']}")
    print(f"Target Churn Rate: {report['target_distribution']['churn_rate_pct']:.2f}%")
    print(f"Train Shape: {report['train_transformed_shape']}, Test Shape: {report['test_transformed_shape']}")
    print(f"Features Match: {report['columns_match']} ({report['transformed_features_count']} features)")
    print(f"Nulls (Train / Test): {report['null_values']['train_nulls']} / {report['null_values']['test_nulls']}")
    print(f"Unseen category handling: {report['unseen_category_handled']}")
    print(f"5-Fold CV verification: {len(report['cv_verification'])} folds successfully verified")
    print(f"ALL CHECKS PASSED: {report['all_checks_passed']}")
