"""Automated tests for Phase 3: Machine Learning Model Development."""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR, ID_COLUMN, TARGET_COLUMN
from src.data_loader import load_and_prepare_data
from src.train import (
    get_model_pipeline_and_param_grid,
    BalancedGradientBoostingClassifier,
)


class TestModelPipelinesAndArtifacts:
    """Tests verifying pipeline construction, artifact persistence, and inference."""

    @pytest.fixture
    def prepared_data(self):
        X, y = load_and_prepare_data()
        return X, y

    def test_pipeline_construction_reuses_preprocessor(self):
        """Verify each model pipeline wraps the standard preprocessor without duplicating logic."""
        for model_name in ["logistic_regression", "random_forest", "gradient_boosting"]:
            pipeline, param_grid = get_model_pipeline_and_param_grid(model_name)
            assert isinstance(pipeline, Pipeline)
            assert "preprocessor" in pipeline.named_steps
            assert "classifier" in pipeline.named_steps
            assert len(param_grid) > 0

    def test_saved_model_artifact_exists_and_loads(self):
        """Verify the trained model artifact exists on disk and is a valid Scikit-Learn Pipeline."""
        model_path = MODELS_DIR / "best_churn_model.joblib"
        assert model_path.exists(), f"Model artifact not found at {model_path}"
        assert model_path.stat().st_size > 1000, "Model artifact file is suspiciously small"

        loaded_pipeline = joblib.load(model_path)
        assert isinstance(loaded_pipeline, Pipeline)
        assert hasattr(loaded_pipeline, "predict")
        assert hasattr(loaded_pipeline, "predict_proba")

    def test_end_to_end_inference_on_raw_features(self, prepared_data):
        """Verify the saved pipeline performs inference directly on raw DataFrames without manual preprocessing."""
        X, _ = prepared_data
        sample_raw = X.head(20).copy()

        model_path = MODELS_DIR / "best_churn_model.joblib"
        pipeline = joblib.load(model_path)

        # Predict classes
        preds = pipeline.predict(sample_raw)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(sample_raw)
        assert set(preds).issubset({0, 1})

        # Predict probabilities
        probas = pipeline.predict_proba(sample_raw)
        assert probas.shape == (len(sample_raw), 2)
        assert (probas >= 0.0).all() and (probas <= 1.0).all()
        # Ensure row probabilities sum to 1.0
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)


class TestModelReportAndEvaluationArtifacts:
    """Tests verifying model comparison reports and plot artifacts."""

    def test_model_comparison_report_validity(self):
        """Verify the JSON report exists, contains all 3 models, and has valid metric scores."""
        report_path = MODELS_DIR / "model_comparison_report.json"
        assert report_path.exists(), f"Comparison report not found at {report_path}"

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "selected_best_model" in data
        assert "model_comparison" in data

        models = data["model_comparison"]
        assert len(models) == 3
        expected_models = {"logistic_regression", "random_forest", "gradient_boosting"}
        assert set(models.keys()) == expected_models

        for name, info in models.items():
            assert "cv_metrics" in info
            assert "test_metrics" in info
            cv_m = info["cv_metrics"]
            test_m = info["test_metrics"]

            # Validate metric ranges
            assert 0.70 <= cv_m["roc_auc"] <= 1.0
            assert 0.50 <= cv_m["pr_auc"] <= 1.0
            assert 0.0 <= cv_m["brier_score"] <= 0.25
            assert 0.70 <= test_m["roc_auc"] <= 1.0
            assert 0.0 <= test_m["recall"] <= 1.0

    def test_markdown_report_exists_and_populated(self):
        """Verify the markdown comparison report exists and contains table content."""
        md_path = MODELS_DIR / "model_comparison_report.md"
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "Random Forest" in content
        assert "Logistic Regression" in content
        assert "Gradient Boosting" in content
        assert "Holdout Test Set Performance" in content

    def test_evaluation_plots_exist(self):
        """Verify all generated evaluation plots exist and are non-empty."""
        plots_dir = MODELS_DIR / "plots"
        expected_plots = [
            plots_dir / "confusion_matrices.png",
            plots_dir / "roc_and_pr_curves.png",
            plots_dir / "calibration_curves.png",
        ]
        for plot_path in expected_plots:
            assert plot_path.exists(), f"Plot not found: {plot_path}"
            assert plot_path.stat().st_size > 5000, f"Plot file {plot_path} is suspiciously small"
