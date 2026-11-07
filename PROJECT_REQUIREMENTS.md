# Project Requirements: Telco Customer Churn Prediction

## 1. Executive Summary & Business Context

### 1.1 Problem Statement
Customer churn poses a significant challenge for telecommunication service providers. Acquiring new customers typically costs 5 to 25 times more than retaining existing ones. The objective of this project is to build an end-to-end Machine Learning pipeline that identifies customers at high risk of churning, uncovers the primary factors driving churn, and enables targeted, proactive retention strategies.

### 1.2 Key Objectives
- **Predictive Accuracy:** Train and evaluate classification models to identify churn with high recall (to capture churners) while maintaining robust precision and ROC-AUC.
- **Explainability:** Provide interpretable feature attributions (e.g., via SHAP) so business teams can understand why a customer is flagged as high risk.
- **Actionable Insights:** Categorize customer churn risk into actionable tiers (Low, Medium, High Risk) and associate customer profiles with targeted retention playbooks.
- **Production Readiness:** Structure modular, clean, reproducible code covering ingestion, preprocessing, modeling, evaluation, and inference.

---

## 2. Dataset Specification

- **File Path:** `data/Telco-Customer-Churn.csv`
- **Total Records:** 7,043 rows
- **Total Features:** 21 columns (20 features + 1 target variable)
- **Target Variable:** `Churn` (Binary: `"Yes"` / `"No"`)

### 2.1 Feature Catalog

| Category | Column Name | Type | Description |
| :--- | :--- | :--- | :--- |
| **Identifier** | `customerID` | String | Unique customer identifier (to be excluded from modeling features) |
| **Demographics** | `gender` | Categorical | Male or Female |
| | `SeniorCitizen` | Binary (0/1) | Whether the customer is a senior citizen |
| | `Partner` | Categorical | Whether the customer has a partner (Yes/No) |
| | `Dependents` | Categorical | Whether the customer has dependents (Yes/No) |
| **Account Info** | `tenure` | Integer | Number of months the customer has stayed with the company |
| | `Contract` | Categorical | The contract term (Month-to-month, One year, Two year) |
| | `PaperlessBilling` | Categorical | Whether customer has paperless billing (Yes/No) |
| | `PaymentMethod` | Categorical | Electronic check, Mailed check, Bank transfer (auto), Credit card (auto) |
| | `MonthlyCharges` | Float | The amount charged to the customer monthly |
| | `TotalCharges` | Float / String | The total amount charged to the customer (contains whitespace strings for new users) |
| **Subscribed Services** | `PhoneService` | Categorical | Whether customer has a phone service (Yes/No) |
| | `MultipleLines` | Categorical | No phone service, No, Yes |
| | `InternetService` | Categorical | DSL, Fiber optic, No |
| | `OnlineSecurity` | Categorical | Yes, No, No internet service |
| | `OnlineBackup` | Categorical | Yes, No, No internet service |
| | `DeviceProtection` | Categorical | Yes, No, No internet service |
| | `TechSupport` | Categorical | Yes, No, No internet service |
| | `StreamingTV` | Categorical | Yes, No, No internet service |
| | `StreamingMovies` | Categorical | Yes, No, No internet service |
| **Target** | `Churn` | Categorical | Whether the customer churned (Yes/No) |

---

## 3. Data Preprocessing & Cleaning Requirements

1. **Whitespace & Missing Value Treatment (`TotalCharges`):**
   - In 11 rows where `tenure == 0`, `TotalCharges` contains empty whitespace strings (`" "`).
   - Coerce `TotalCharges` to numeric (`pd.to_numeric(..., errors='coerce')`).
   - Impute missing values with `0` (or `MonthlyCharges * tenure`).
2. **Encoding Strategy:**
   - **Target Variable:** Map `Churn` `"Yes"` -> `1`, `"No"` -> `0`.
   - **Binary Features:** Map `"Yes"`/`"No"` to `1`/`0`.
   - **Multi-Class Categoricals:** Apply One-Hot Encoding (`OneHotEncoder(drop='first', sparse_output=False)`).
3. **Scaling:**
   - Standardize or min-max scale continuous numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) for distance-based models (Logistic Regression, SVM, Neural Nets). Tree-based models (Random Forest, LightGBM, XGBoost) can accept raw numeric values.
4. **Data Leakage Prevention:**
   - Fit all encoders, scalers, and imputers strictly on training folds/splits and transform validation/test sets accordingly.

---

## 4. Modeling & Evaluation Requirements

### 4.1 Candidate Models
1. **Baseline:** Logistic Regression with L2 regularization.
2. **Tree Ensembles:** Random Forest Classifier.
3. **Gradient Boosting:** LightGBM / XGBoost / CatBoost.
4. **Optional:** Neural Network (MLP) or Stacking Ensemble.

### 4.2 Cross-Validation Strategy
- **Stratified K-Fold CV** (5 folds recommended) to preserve churn class distribution (~26.5% positive class).

### 4.3 Evaluation Metrics
- **Primary Metric:** **ROC-AUC** & **PR-AUC (Average Precision)** — invariant to threshold shifts and class imbalance.
- **Secondary Metrics:**
  - **Recall (Sensitivity):** Prioritizing capturing maximum true churners.
  - **F1-Score / Precision:** Balancing cost of unnecessary retention incentives against lost revenue.
  - **Brier Score / Calibration Curve:** Ensuring predicted probabilities reflect true empirical risk.

---

## 5. Model Interpretability & Business Strategy

1. **Global Interpretability:**
   - Feature importance rankings (MDI, Permutation Importance, and SHAP summary plots).
   - Quantify key churn drivers (e.g., month-to-month contracts, fiber optic pricing, lack of tech support).
2. **Local Interpretability:**
   - Individual SHAP waterfall or force plots explaining risk scores for specific accounts.
3. **Customer Segmentation & Tiers:**
   - Segment customers into:
     - **Low Risk:** Churn probability < 0.30
     - **Medium Risk:** Churn probability 0.30 – 0.60
     - **High Risk:** Churn probability > 0.60
   - Prioritize high-risk accounts with high `MonthlyCharges` for personalized retention offers.

---

## 6. Proposed Project Architecture

```
customer-churn-prediction/
│
├── data/
│   ├── Telco-Customer-Churn.csv       # Raw dataset
│   └── processed/                     # Cleaned / engineered datasets (optional)
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_model_experimentation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Ingestion and type conversions
│   ├── preprocessor.py                # Pipeline transformations & encoding
│   ├── train.py                       # Cross-validation and model training
│   ├── evaluate.py                    # Metrics, plots, and evaluation reports
│   └── predict.py                     # Inference module for new customer records
│
├── models/                            # Saved trained model artifacts (.joblib / .pkl)
├── requirements.txt                   # Dependency list
├── PROJECT_REQUIREMENTS.md            # This specification file
└── README.md                          # Project documentation and run guide
```

---

## 7. Next Steps & Deliverables

1. [x] Ingest & verify raw data in `data/Telco-Customer-Churn.csv`.
2. [x] Formulate foundational `PROJECT_REQUIREMENTS.md`.
3. [ ] Exploratory Data Analysis (EDA) script or notebook.
4. [ ] Build reusable preprocessing and feature engineering pipelines.
5. [ ] Train and benchmark candidate machine learning algorithms.
6. [ ] Generate model evaluation reports (ROC curves, confusion matrices, SHAP analysis).
7. [ ] Deploy inference script or interactive web dashboard (Streamlit / FastAPI).
