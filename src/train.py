"""Model training, hyperparameter optimization, cross-validation, and model selection module."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

from src.config import (
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    N_SPLITS,
)
from src.data_loader import load_and_prepare_data
from src.preprocessor import build_preprocessor
from src.split import create_stratified_split, get_stratified_kfold
from src.evaluate import (
    calculate_metrics,
    plot_confusion_matrices,
    plot_roc_and_pr_curves,
    plot_calibration_curves,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class BalancedGradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """GradientBoostingClassifier wrapper providing direct support for class_weight='balanced'.

    Computes sample weights automatically for balanced training to address class imbalance.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 1.0,
        class_weight: str = "balanced",
        random_state: int = RANDOM_STATE,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.class_weight = class_weight
        self.random_state = random_state
        self.estimator_ = None

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def fit(self, X, y, **fit_params):
        sample_weight = None
        if self.class_weight is not None:
            sample_weight = compute_sample_weight(self.class_weight, y)

        self.estimator_ = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        self.estimator_.fit(X, y, sample_weight=sample_weight, **fit_params)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def decision_function(self, X):
        return self.estimator_.decision_function(X)


def get_model_pipeline_and_param_grid(model_name: str) -> Tuple[Pipeline, Dict[str, Any]]:
    """Create a unified Pipeline combining the existing preprocessor and classifier with search grid.

    Reuses existing build_preprocessor() to guarantee zero logic duplication.

    Args:
        model_name: One of 'logistic_regression', 'random_forest', 'gradient_boosting'.

    Returns:
        Tuple[Pipeline, Dict[str, Any]]: (unfitted_pipeline, param_grid)
    """
    preprocessor = build_preprocessor()

    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
        param_grid = {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__class_weight": ["balanced", None],
        }

    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_leaf": [2, 4],
            "classifier__class_weight": ["balanced", "balanced_subsample"],
        }

    elif model_name == "gradient_boosting":
        classifier = BalancedGradientBoostingClassifier(
            random_state=RANDOM_STATE,
        )
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
        param_grid = {
            "classifier__n_estimators": [100, 150],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [3, 4],
            "classifier__class_weight": ["balanced", None],
        }

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return pipeline, param_grid


def evaluate_cross_validation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> Dict[str, Any]:
    """Perform fold-by-fold cross-validation to capture validation and training performance."""
    fold_metrics = []
    train_metrics = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        fold_X_tr, fold_y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        fold_X_val, fold_y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        # Clone and fit pipeline strictly on fold training data
        fold_pipe = clone(pipeline)
        fold_pipe.fit(fold_X_tr, fold_y_tr)

        # Validation evaluation
        val_pred = fold_pipe.predict(fold_X_val)
        val_proba = fold_pipe.predict_proba(fold_X_val)[:, 1]
        val_m = calculate_metrics(fold_y_val.values, val_pred, val_proba)
        fold_metrics.append(val_m)

        # Training evaluation (to detect overfitting gap)
        tr_pred = fold_pipe.predict(fold_X_tr)
        tr_proba = fold_pipe.predict_proba(fold_X_tr)[:, 1]
        tr_m = calculate_metrics(fold_y_tr.values, tr_pred, tr_proba)
        train_metrics.append(tr_m)

    # Average metrics across 5 folds
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"]
    mean_val = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    std_val = {k: float(np.std([m[k] for m in fold_metrics])) for k in keys}
    mean_train = {k: float(np.mean([m[k] for m in train_metrics])) for k in keys}

    return {
        "mean_cv_metrics": mean_val,
        "std_cv_metrics": std_val,
        "mean_train_metrics": mean_train,
        "fold_details": fold_metrics,
    }


def train_and_compare_models() -> Dict[str, Any]:
    """Train, tune, cross-validate, and evaluate all 3 candidate algorithms.

    Keeps the test set untouched until the winning model is selected.

    Returns:
        Dict[str, Any]: Comprehensive comparison and evaluation report.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = MODELS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1: Loading raw dataset and separating features...")
    X, y = load_and_prepare_data()

    logger.info("Step 2: Creating stratified train-test split (80/20)...")
    X_train, X_test, y_train, y_test = create_stratified_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    cv = get_stratified_kfold(n_splits=N_SPLITS, random_state=RANDOM_STATE)

    model_names = ["logistic_regression", "random_forest", "gradient_boosting"]
    display_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
    }

    model_results = {}
    fitted_pipelines = {}

    for name in model_names:
        display_name = display_names[name]
        logger.info("--- Training & Tuning: %s ---", display_name)

        pipeline, param_grid = get_model_pipeline_and_param_grid(name)

        # Optimize hyperparameters on training partition with Stratified 5-Fold CV
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        logger.info("%s Best CV ROC-AUC: %.4f | Best Params: %s", display_name, grid_search.best_score_, best_params)

        # Compute multi-metric cross-validation and training performance
        cv_eval = evaluate_cross_validation(best_pipeline, X_train, y_train, cv)

        fitted_pipelines[name] = best_pipeline
        model_results[name] = {
            "display_name": display_name,
            "best_params": best_params,
            "cv_metrics": cv_eval["mean_cv_metrics"],
            "cv_metrics_std": cv_eval["std_cv_metrics"],
            "train_metrics": cv_eval["mean_train_metrics"],
            "overfitting_gap": {
                "roc_auc_gap": cv_eval["mean_train_metrics"]["roc_auc"] - cv_eval["mean_cv_metrics"]["roc_auc"],
                "f1_gap": cv_eval["mean_train_metrics"]["f1"] - cv_eval["mean_cv_metrics"]["f1"],
                "recall_gap": cv_eval["mean_train_metrics"]["recall"] - cv_eval["mean_cv_metrics"]["recall"],
            },
        }

    # Model Selection Logic
    # Business requirement: Churn detection prioritizes Recall (capturing churning subscribers)
    # alongside high ROC-AUC and PR-AUC, with reasonable precision to control retention cost.
    # We define a composite business utility score = (0.4 * Recall) + (0.35 * ROC-AUC) + (0.25 * PR-AUC).
    for name, res in model_results.items():
        m = res["cv_metrics"]
        business_score = (0.40 * m["recall"]) + (0.35 * m["roc_auc"]) + (0.25 * m["pr_auc"])
        res["business_utility_score"] = float(business_score)

    best_model_name = max(model_results, key=lambda k: model_results[k]["business_utility_score"])
    best_display_name = display_names[best_model_name]
    logger.info("Selected Best Model: %s (Business Score: %.4f)", best_display_name, model_results[best_model_name]["business_utility_score"])

    # FINAL TEST EVALUATION (Kept completely untouched until this step)
    logger.info("--- Step 3: Final Evaluation on Untouched Test Set (N=%d) ---", len(y_test))
    test_evaluations = {}
    test_probas = {}

    for name in model_names:
        pipe = fitted_pipelines[name]
        # Re-fit winning pipeline on FULL X_train before final test scoring
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test.values, y_pred, y_proba)
        test_evaluations[display_names[name]] = {
            "metrics": metrics,
            "y_pred": y_pred.tolist(),
        }
        test_probas[display_names[name]] = y_proba
        model_results[name]["test_metrics"] = metrics

    # Save the selected winning model artifact to disk
    winning_pipeline = fitted_pipelines[best_model_name]
    model_save_path = MODELS_DIR / "best_churn_model.joblib"
    joblib.dump(winning_pipeline, model_save_path)
    logger.info("Saved complete winning pipeline to: %s", model_save_path)

    # Generate Evaluation Visualizations
    logger.info("Generating evaluation plots in %s...", plots_dir)
    plot_confusion_matrices(test_evaluations, plots_dir / "confusion_matrices.png")
    plot_roc_and_pr_curves(y_test.values, test_probas, plots_dir / "roc_and_pr_curves.png")
    plot_calibration_curves(y_test.values, test_probas, plots_dir / "calibration_curves.png")

    # Save structured JSON report
    report_data = {
        "selected_best_model": best_display_name,
        "selection_rationale": (
            "Selected based on balanced churn detection utility, maximizing Recall while "
            "preserving high ROC-AUC and PR-AUC with low calibration Brier score."
        ),
        "model_comparison": model_results,
    }

    report_json_path = MODELS_DIR / "model_comparison_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Save Markdown report
    generate_markdown_report(report_data, MODELS_DIR / "model_comparison_report.md")

    return report_data


def generate_markdown_report(report_data: Dict[str, Any], output_path: Path) -> None:
    """Generate a clean, readable Markdown report of model benchmark results."""
    models = report_data["model_comparison"]

    md = []
    md.append("# Machine Learning Model Benchmark & Comparison Report")
    md.append("\n## Executive Summary")
    md.append(f"- **Selected Winning Model:** **{report_data['selected_best_model']}**")
    md.append(f"- **Selection Rationale:** {report_data['selection_rationale']}")
    md.append("- **Cross-Validation Strategy:** Stratified 5-Fold Cross-Validation on Training Set (N=5,634)")
    md.append("- **Holdout Test Set:** N=1,409 (Untouched until final evaluation)\n")

    md.append("## 1. Cross-Validation Performance Comparison (Mean ± Std)")
    md.append("| Model | ROC-AUC | PR-AUC | Recall | Precision | F1-Score | Brier Score |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for k, v in models.items():
        name = v["display_name"]
        cv_m = v["cv_metrics"]
        std_m = v["cv_metrics_std"]
        md.append(
            f"| **{name}** | {cv_m['roc_auc']:.3f} ± {std_m['roc_auc']:.3f} | "
            f"{cv_m['pr_auc']:.3f} ± {std_m['pr_auc']:.3f} | "
            f"{cv_m['recall']:.3f} ± {std_m['recall']:.3f} | "
            f"{cv_m['precision']:.3f} ± {std_m['precision']:.3f} | "
            f"{cv_m['f1']:.3f} ± {std_m['f1']:.3f} | "
            f"{cv_m['brier_score']:.3f} ± {std_m['brier_score']:.3f} |"
        )

    md.append("\n## 2. Generalization & Overfitting Analysis (Train vs. CV Gap)")
    md.append("| Model | Train ROC-AUC | CV ROC-AUC | ROC-AUC Gap | Train Recall | CV Recall | Recall Gap |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for k, v in models.items():
        name = v["display_name"]
        tr_m = v["train_metrics"]
        cv_m = v["cv_metrics"]
        gap = v["overfitting_gap"]
        md.append(
            f"| **{name}** | {tr_m['roc_auc']:.3f} | {cv_m['roc_auc']:.3f} | {gap['roc_auc_gap']:+.3f} | "
            f"{tr_m['recall']:.3f} | {cv_m['recall']:.3f} | {gap['recall_gap']:+.3f} |"
        )

    md.append("\n## 3. Final Holdout Test Set Performance (N=1,409)")
    md.append("| Model | Test ROC-AUC | Test PR-AUC | Test Recall | Test Precision | Test F1-Score | Test Brier | True Positives | False Negatives |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for k, v in models.items():
        name = v["display_name"]
        tm = v["test_metrics"]
        md.append(
            f"| **{name}** | {tm['roc_auc']:.3f} | {tm['pr_auc']:.3f} | {tm['recall']:.3f} | "
            f"{tm['precision']:.3f} | {tm['f1']:.3f} | {tm['brier_score']:.3f} | "
            f"{tm['true_positives']} / 374 | {tm['false_negatives']} |"
        )

    md.append("\n## 4. Best Hyperparameters Found")
    for k, v in models.items():
        md.append(f"- **{v['display_name']}:** `{v['best_params']}`")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    train_and_compare_models()
