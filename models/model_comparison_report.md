# Machine Learning Model Benchmark & Comparison Report

## Executive Summary
- **Selected Winning Model:** **Random Forest**
- **Selection Rationale:** Selected based on balanced churn detection utility, maximizing Recall while preserving high ROC-AUC and PR-AUC with low calibration Brier score.
- **Cross-Validation Strategy:** Stratified 5-Fold Cross-Validation on Training Set (N=5,634)
- **Holdout Test Set:** N=1,409 (Untouched until final evaluation)

## 1. Cross-Validation Performance Comparison (Mean ± Std)
| Model | ROC-AUC | PR-AUC | Recall | Precision | F1-Score | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.846 ± 0.013 | 0.661 ± 0.019 | 0.549 ± 0.039 | 0.656 ± 0.028 | 0.597 ± 0.030 | 0.135 ± 0.005 |
| **Random Forest** | 0.846 ± 0.009 | 0.658 ± 0.018 | 0.780 ± 0.019 | 0.537 ± 0.014 | 0.636 ± 0.012 | 0.157 ± 0.004 |
| **Gradient Boosting** | 0.849 ± 0.011 | 0.672 ± 0.021 | 0.515 ± 0.033 | 0.670 ± 0.026 | 0.582 ± 0.027 | 0.134 ± 0.005 |

## 2. Generalization & Overfitting Analysis (Train vs. CV Gap)
| Model | Train ROC-AUC | CV ROC-AUC | ROC-AUC Gap | Train Recall | CV Recall | Recall Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.850 | 0.846 | +0.004 | 0.553 | 0.549 | +0.004 |
| **Random Forest** | 0.927 | 0.846 | +0.080 | 0.906 | 0.780 | +0.126 |
| **Gradient Boosting** | 0.873 | 0.849 | +0.024 | 0.547 | 0.515 | +0.032 |

## 3. Final Holdout Test Set Performance (N=1,409)
| Model | Test ROC-AUC | Test PR-AUC | Test Recall | Test Precision | Test F1-Score | Test Brier | True Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.841 | 0.628 | 0.556 | 0.654 | 0.601 | 0.139 | 208 / 374 | 166 |
| **Random Forest** | 0.843 | 0.657 | 0.783 | 0.530 | 0.632 | 0.160 | 293 / 374 | 81 |
| **Gradient Boosting** | 0.845 | 0.661 | 0.508 | 0.674 | 0.579 | 0.136 | 190 / 374 | 184 |

## 4. Best Hyperparameters Found
- **Logistic Regression:** `{'classifier__C': 10.0, 'classifier__class_weight': None}`
- **Random Forest:** `{'classifier__class_weight': 'balanced', 'classifier__max_depth': 10, 'classifier__min_samples_leaf': 4, 'classifier__n_estimators': 200}`
- **Gradient Boosting:** `{'classifier__class_weight': None, 'classifier__learning_rate': 0.05, 'classifier__max_depth': 3, 'classifier__n_estimators': 100}`