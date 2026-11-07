"""Evaluation metrics computation, probability calibration, and visualization generators."""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive classification and calibration metrics.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_proba: Predicted probabilities for positive class (Churn = 1).

    Returns:
        Dict[str, float]: Computed metrics dictionary.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def plot_confusion_matrices(
    results: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot and save confusion matrices for all evaluated models side-by-side."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, res) in zip(axes, results.items()):
        cm = np.array([
            [res["metrics"]["true_negatives"], res["metrics"]["false_positives"]],
            [res["metrics"]["false_negatives"], res["metrics"]["true_positives"]],
        ])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
        )
        ax.set_title(f"{model_name}\nConfusion Matrix", fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_and_pr_curves(
    y_test: np.ndarray,
    model_probas: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Plot and save comparative ROC and Precision-Recall curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    # ROC Curves
    for (name, proba), color in zip(model_probas.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, proba)
        score = roc_auc_score(y_test, proba)
        axes[0].plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {score:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7, label="Chance Baseline (AUC = 0.500)")
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall / Sensitivity)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # PR Curves
    base_rate = np.mean(y_test)
    for (name, proba), color in zip(model_probas.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, proba)
        score = average_precision_score(y_test, proba)
        axes[1].plot(rec, prec, color=color, lw=2, label=f"{name} (PR-AUC = {score:.3f})")

    axes[1].axhline(base_rate, color="k", linestyle="--", lw=1, alpha=0.7, label=f"Chance Baseline ({base_rate:.3f})")
    axes[1].set_title("Precision-Recall (PR) Curve", fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_curves(
    y_test: np.ndarray,
    model_probas: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Plot and save probability calibration reliability curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    # Reliability Diagram
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    for (name, proba), color in zip(model_probas.items(), colors):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, proba, n_bins=10, strategy="uniform"
        )
        brier = brier_score_loss(y_test, proba)
        axes[0].plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            color=color,
            label=f"{name} (Brier = {brier:.3f})",
        )

    axes[0].set_title("Calibration Curve (Reliability Diagram)", fontweight="bold")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Empirical Fraction of Positives")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Predicted Probability Histograms
    for (name, proba), color in zip(model_probas.items(), colors):
        axes[1].hist(
            proba,
            range=(0, 1),
            bins=20,
            histtype="step",
            lw=2,
            color=color,
            label=name,
        )

    axes[1].set_title("Predicted Probability Distribution", fontweight="bold")
    axes[1].set_xlabel("Predicted Churn Probability")
    axes[1].set_ylabel("Customer Count")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
