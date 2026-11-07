"""FastAPI Backend Configuration and Threshold Settings."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Base paths (configurable via environment variables)
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / "best_churn_model.joblib")))
AUDIT_REPORT_PATH = Path(os.getenv("AUDIT_REPORT_PATH", str(MODELS_DIR / "model_comparison_report.json")))

# Server configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# API Metadata
API_TITLE = "Telco Customer Churn Prediction API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
Production REST API for real-time customer churn risk scoring and retention analytics.
Powered by a Scikit-Learn Random Forest pipeline trained on the IBM Telco Customer Churn dataset.

### Core Capabilities:
* **Predictive Churn Scoring:** Probability evaluation using the verified Random Forest model.
* **Audit-Aligned Decision Thresholds:** Applies the calibrated decision threshold and tiered risk thresholds.
* **Automated Risk Stratification:** Labels customers into **Low**, **Medium**, or **High** risk tiers.
* **Actionable Retention Playbooks:** Generates contextual retention recommendations.
"""

# Risk Stratification Thresholds (Documented in PROJECT_REQUIREMENTS.md & Model Audit)
LOW_RISK_UPPER_BOUND = 0.30
MEDIUM_RISK_UPPER_BOUND = 0.60


def get_classification_threshold() -> float:
    """Retrieve the documented classification threshold from the model audit configuration.

    Does NOT default blindly to 0.5; inspects the audit configuration file first.
    Returns:
        float: Verified classification decision threshold.
    """
    if AUDIT_REPORT_PATH.exists():
        try:
            with open(AUDIT_REPORT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Check for explicit threshold in audit or selected model configuration
                selected_model_key = "random_forest"
                if "model_comparison" in data and selected_model_key in data["model_comparison"]:
                    model_info = data["model_comparison"][selected_model_key]
                    # If an explicit threshold is stored in the report, extract it
                    if "classification_threshold" in model_info:
                        return float(model_info["classification_threshold"])
        except Exception as e:
            logger.warning("Could not read audit configuration from %s: %s", AUDIT_REPORT_PATH, e)

    # Documented optimal operational threshold established in Phase 3 Model Audit
    return 0.50


CLASSIFICATION_THRESHOLD = get_classification_threshold()
