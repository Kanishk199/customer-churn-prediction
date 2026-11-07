"""Comprehensive unit and integration tests for Phase 2 data preprocessing pipeline."""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.config import (
    RAW_DATA_PATH,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    N_SPLITS,
)
from src.data_loader import (
    load_raw_data,
    clean_dataset,
    separate_features_target,
    load_and_prepare_data,
)
from src.preprocessor import (
    build_preprocessor,
    get_feature_names,
    transform_to_dataframe,
)
from src.split import create_stratified_split, get_stratified_kfold, summarize_split
from src.pipeline import create_feature_pipeline, fit_and_transform_split


class TestDataLoadingAndCleaning:
    """Tests for raw data integrity, type coercion, and cleaning."""

    def test_raw_data_loads_and_is_unmodified(self):
        """Verify the raw dataset loads correctly with expected columns and rows."""
        df = load_raw_data()
        assert df.shape == (7043, 21), f"Expected shape (7043, 21), got {df.shape}"
        assert ID_COLUMN in df.columns
        assert TARGET_COLUMN in df.columns

    def test_clean_dataset_converts_totalcharges_and_fills_missing(self):
        """Verify TotalCharges whitespace strings are converted to float and imputed with 0.0."""
        raw_df = load_raw_data()
        cleaned_df = clean_dataset(raw_df)

        # Check type
        assert np.issubdtype(cleaned_df["TotalCharges"].dtype, np.number), (
            f"Expected numeric dtype for TotalCharges, got {cleaned_df['TotalCharges'].dtype}"
        )
        # Check no missing values remain
        assert cleaned_df["TotalCharges"].isnull().sum() == 0

        # Check records with tenure == 0 have TotalCharges == 0.0
        zero_tenure_mask = cleaned_df["tenure"] == 0
        assert (cleaned_df.loc[zero_tenure_mask, "TotalCharges"] == 0.0).all()

    def test_separate_features_target_removes_id_and_encodes_churn(self):
        """Verify customerID is dropped from features and Churn is binary 0/1."""
        raw_df = load_raw_data()
        clean_df = clean_dataset(raw_df)
        X, y = separate_features_target(clean_df)

        assert ID_COLUMN not in X.columns, f"{ID_COLUMN} must be excluded from features"
        assert TARGET_COLUMN not in X.columns, f"{TARGET_COLUMN} must be excluded from features"
        assert X.shape[1] == 19, f"Expected 19 features, got {X.shape[1]}"
        assert set(y.unique()) == {0, 1}, f"Target values must be binary {{0, 1}}, got {set(y.unique())}"
        assert len(X) == len(y) == 7043


class TestStratifiedSplitting:
    """Tests for stratified train/test partitioning."""

    def test_stratified_split_preserves_class_proportions(self):
        """Verify train and test partitions maintain the ~26.5% minority churn ratio."""
        X, y = load_and_prepare_data()
        X_train, X_test, y_train, y_test = create_stratified_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        assert len(X_train) == 5634
        assert len(X_test) == 1409

        overall_churn = y.mean() * 100
        train_churn = y_train.mean() * 100
        test_churn = y_test.mean() * 100

        # Discrepancy between splits and overall should be negligible (< 0.2%)
        assert abs(train_churn - overall_churn) < 0.2
        assert abs(test_churn - overall_churn) < 0.2
        assert abs(train_churn - test_churn) < 0.2

        # Ensure no index leakage
        assert set(X_train.index).isdisjoint(set(X_test.index))


class TestPreprocessingPipeline:
    """Tests for ColumnTransformer, Pipeline, and feature transformations."""

    @pytest.fixture
    def split_data(self):
        X, y = load_and_prepare_data()
        return create_stratified_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    def test_pipeline_fitted_strictly_on_train_transforms_both(self, split_data):
        """Ensure preprocessor fits only on train data and transforms both partitions."""
        X_train, X_test, y_train, y_test = split_data
        preprocessor, X_train_trans, X_test_trans = fit_and_transform_split(X_train, X_test)

        assert isinstance(preprocessor, ColumnTransformer)
        assert X_train_trans.shape[0] == len(X_train)
        assert X_test_trans.shape[0] == len(X_test)
        assert X_train_trans.shape[1] == X_test_trans.shape[1]

        # Verify exact column alignment
        assert list(X_train_trans.columns) == list(X_test_trans.columns)

    def test_transformed_data_has_no_nans_or_infs(self, split_data):
        """Ensure transformed features contain zero missing or infinite values."""
        X_train, X_test, _, _ = split_data
        _, X_train_trans, X_test_trans = fit_and_transform_split(X_train, X_test)

        assert X_train_trans.isnull().sum().sum() == 0
        assert X_test_trans.isnull().sum().sum() == 0
        assert np.isfinite(X_train_trans.values).all()
        assert np.isfinite(X_test_trans.values).all()

    def test_standard_scaler_centers_and_scales_train(self, split_data):
        """Verify numerical features in training split have mean ~ 0 and std ~ 1."""
        X_train, X_test, _, _ = split_data
        _, X_train_trans, _ = fit_and_transform_split(X_train, X_test)

        for col in NUMERICAL_FEATURES:
            mean_val = X_train_trans[col].mean()
            std_val = X_train_trans[col].std()
            assert abs(mean_val) < 1e-4, f"Column {col} mean={mean_val} not centered around 0"
            assert abs(std_val - 1.0) < 1e-2, f"Column {col} std={std_val} not scaled to ~1.0"

    def test_onehotencoder_handles_unseen_categories(self, split_data):
        """Verify handle_unknown='ignore' handles novel category values without errors or NaNs."""
        X_train, X_test, _, _ = split_data
        preprocessor = build_preprocessor()
        preprocessor.fit(X_train)

        # Inject an unseen category into test data
        corrupted_test = X_test.copy().head(10)
        corrupted_test["Contract"] = "Decade-to-decade"
        corrupted_test["PaymentMethod"] = "Cryptocurrency"

        # Transform should succeed without throwing error
        transformed = transform_to_dataframe(preprocessor, corrupted_test)

        assert transformed.shape[1] == len(get_feature_names(preprocessor))
        assert transformed.isnull().sum().sum() == 0
        assert np.isfinite(transformed.values).all()

    def test_scikit_learn_pipeline_wrapper(self, split_data):
        """Verify create_feature_pipeline() constructs a valid scikit-learn Pipeline."""
        X_train, X_test, y_train, _ = split_data
        pipe = create_feature_pipeline()
        assert isinstance(pipe, Pipeline)

        pipe.fit(X_train, y_train)
        transformed = pipe.transform(X_test)
        assert transformed.shape == (len(X_test), 46)


class TestCrossValidationPreparation:
    """Tests verifying readiness for 5-fold cross-validation."""

    def test_stratified_5_fold_cv_splits_and_leakage_free_fitting(self):
        """Ensure 5-fold cross-validation executes on training data with zero leakage."""
        X, y = load_and_prepare_data()
        X_train, _, y_train, _ = create_stratified_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        cv = get_stratified_kfold(n_splits=N_SPLITS, random_state=RANDOM_STATE)
        fold_count = 0

        for train_idx, val_idx in cv.split(X_train, y_train):
            fold_count += 1
            fold_X_tr, fold_y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            fold_X_val, fold_y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            # Fit preprocessor exclusively on fold training data
            fold_prep = build_preprocessor()
            fold_prep.fit(fold_X_tr)

            tr_trans = fold_prep.transform(fold_X_tr)
            val_trans = fold_prep.transform(fold_X_val)

            assert tr_trans.shape[1] == val_trans.shape[1] == 46
            assert np.isnan(tr_trans).sum() == 0
            assert np.isnan(val_trans).sum() == 0

            # Check fold stratification
            val_churn_pct = fold_y_val.mean() * 100
            assert abs(val_churn_pct - (y_train.mean() * 100)) < 0.5

        assert fold_count == 5
