"""Inference service wrapping the saved Scikit-Learn Random Forest pipeline."""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import HTTPException, status
from sklearn.pipeline import Pipeline

from api.config import (
    MODEL_PATH,
    AUDIT_REPORT_PATH,
    CLASSIFICATION_THRESHOLD,
    LOW_RISK_UPPER_BOUND,
    MEDIUM_RISK_UPPER_BOUND,
)
from api.schemas import (
    CustomerInput,
    PredictionResponse,
    LocalExplanation,
    FeatureContribution,
    ExplanationResponse,
)

logger = logging.getLogger(__name__)

# Required column sequence matching the original preprocessor expectations
ORDERED_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


class ModelService:
    """Manages model loading, health diagnostics, local SHAP explainability, and inference execution."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or MODEL_PATH
        self.pipeline: Optional[Pipeline] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: Optional[List[str]] = None
        self.base_value: Optional[float] = None
        self.is_loaded: bool = False
        self.load_error: Optional[str] = None

    def load_model(self) -> None:
        """Load the persisted joblib model pipeline from disk and initialize TreeExplainer.

        CRITICAL: Never retrains or modifies the model.
        """
        if not self.model_path.exists():
            self.is_loaded = False
            self.load_error = f"Model artifact not found at {self.model_path}"
            logger.error(self.load_error)
            return

        try:
            loaded_artifact = joblib.load(self.model_path)
            if not isinstance(loaded_artifact, Pipeline):
                raise TypeError(f"Expected Scikit-Learn Pipeline, got {type(loaded_artifact)}")

            self.pipeline = loaded_artifact

            # Extract preprocessor and classifier for SHAP explainability
            preprocessor = self.pipeline.named_steps.get("preprocessor")
            classifier = self.pipeline.named_steps.get("classifier")

            if preprocessor is not None and classifier is not None:
                self.feature_names = list(preprocessor.get_feature_names_out())
                self.explainer = shap.TreeExplainer(classifier)

                ev = self.explainer.expected_value
                if isinstance(ev, (list, np.ndarray)):
                    self.base_value = float(ev[1])
                else:
                    self.base_value = float(ev)

                logger.info(
                    "TreeExplainer initialized for %s with %d features (base probability: %.4f)",
                    type(classifier).__name__,
                    len(self.feature_names),
                    self.base_value,
                )

            self.is_loaded = True
            self.load_error = None
            logger.info("Successfully loaded model pipeline from %s", self.model_path)
        except Exception as exc:
            self.is_loaded = False
            self.load_error = f"Failed to deserialize model or initialize explainer: {str(exc)}"
            logger.exception(self.load_error)

    def determine_risk_tier(self, probability: float) -> str:
        """Classify churn probability into documented business risk tiers.

        - Low Risk:    P < 0.30
        - Medium Risk: 0.30 <= P < 0.60
        - High Risk:   P >= 0.60
        """
        if probability < LOW_RISK_UPPER_BOUND:
            return "Low Risk"
        elif probability < MEDIUM_RISK_UPPER_BOUND:
            return "Medium Risk"
        else:
            return "High Risk"

    def generate_recommendation(self, probability: float, risk_tier: str, customer: CustomerInput) -> str:
        """Prescribe targeted business actions based on customer risk and drivers."""
        recs = []

        if risk_tier == "High Risk":
            recs.append("Urgent retention intervention required.")
            if customer.Contract == "Month-to-month":
                recs.append("Incentivize migration to a 1-year or 2-year contract with a discounted renewal rate.")
            if customer.PaymentMethod == "Electronic check":
                recs.append("Encourage automated payment setup (credit card or bank transfer) with a one-time bill credit.")
            if customer.TechSupport == "No" or customer.OnlineSecurity == "No":
                recs.append("Offer complimentary 3-month trial of TechSupport and OnlineSecurity to reduce attrition friction.")
        elif risk_tier == "Medium Risk":
            recs.append("Proactive account engagement advised.")
            if customer.tenure <= 6:
                recs.append("Customer is in the vulnerable onboarding window (tenure <= 6); trigger dedicated customer success check-in.")
            recs.append("Send satisfaction survey and verify service quality.")
        else:
            recs.append("Account is stable. Maintain standard automated communications and identify cross-selling opportunities.")

        return " ".join(recs)

    def _describe_feature_contribution(
        self,
        feature_name: str,
        feature_val: float,
        shap_val: float,
        customer: CustomerInput,
    ) -> str:
        """Provide intuitive, human-readable description for feature contribution."""
        sign_str = f"+{shap_val * 100:.1f}%" if shap_val > 0 else f"{shap_val * 100:.1f}%"

        if "Contract_Month-to-month" in feature_name and feature_val == 1.0:
            return f"Month-to-month contract flexibility strongly increases churn hazard ({sign_str})."
        elif "Contract_Two year" in feature_name and feature_val == 1.0:
            return f"Two-year contract commitment anchors long-term customer loyalty ({sign_str})."
        elif "Contract_One year" in feature_name and feature_val == 1.0:
            return f"One-year contract provides stability against immediate cancellation ({sign_str})."
        elif feature_name == "tenure":
            if customer.tenure <= 6:
                return f"Customer is in the vulnerable onboarding cliff ({customer.tenure} months tenure) ({sign_str})."
            else:
                return f"Customer has established tenure of {customer.tenure} months ({sign_str})."
        elif "InternetService_Fiber optic" in feature_name and feature_val == 1.0:
            return f"Fiber optic service with premium charges correlates with higher churn sensitivity ({sign_str})."
        elif "InternetService_DSL" in feature_name and feature_val == 1.0:
            return f"DSL service exhibits stable subscriber retention ({sign_str})."
        elif "PaymentMethod_Electronic check" in feature_name and feature_val == 1.0:
            return f"Manual Electronic Check payment friction increases attrition risk ({sign_str})."
        elif "OnlineSecurity_No" in feature_name and feature_val == 1.0:
            return f"Lack of OnlineSecurity add-on elevates account churn vulnerability ({sign_str})."
        elif "TechSupport_No" in feature_name and feature_val == 1.0:
            return f"Absence of dedicated TechSupport increases attrition probability ({sign_str})."
        elif feature_name == "MonthlyCharges":
            return f"Monthly billing amount of ${customer.MonthlyCharges:.2f} impacts price sensitivity ({sign_str})."
        elif feature_name == "TotalCharges":
            return f"Cumulative lifetime spend of ${customer.TotalCharges:.2f} reflects account investment ({sign_str})."
        else:
            action = "increases churn risk" if shap_val > 0 else "supports customer retention"
            return f"{feature_name} (value: {feature_val:.1f}) {action} by {sign_str}."

    def explain_customer(
        self,
        df_row: pd.DataFrame,
        churn_prob: float,
        customer: CustomerInput,
    ) -> LocalExplanation:
        """Compute local SHAP feature attributions explaining this specific customer prediction."""
        if self.explainer is None or self.pipeline is None or self.feature_names is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SHAP explainer engine is not available.",
            )

        preprocessor = self.pipeline.named_steps["preprocessor"]
        X_trans = preprocessor.transform(df_row)
        feature_values = X_trans[0]

        # Calculate Tree SHAP values
        sv = self.explainer.shap_values(X_trans)
        if isinstance(sv, list):
            churn_sv = sv[1][0]
        elif len(sv.shape) == 3:
            churn_sv = sv[0, :, 1]
        else:
            churn_sv = sv[0]

        contributions: List[FeatureContribution] = []
        for feat_name, feat_val, s_val in zip(self.feature_names, feature_values, churn_sv):
            s_float = float(s_val)
            impact = "increases_churn" if s_float > 0 else "decreases_churn"
            desc = self._describe_feature_contribution(feat_name, float(feat_val), s_float, customer)

            contributions.append(
                FeatureContribution(
                    feature=feat_name,
                    feature_value=round(float(feat_val), 4),
                    shap_value=round(s_float, 4),
                    impact=impact,
                    description=desc,
                )
            )

        # Top 5 features increasing churn probability (positive SHAP, descending)
        pos_contribs = sorted([c for c in contributions if c.shap_value > 0], key=lambda x: x.shap_value, reverse=True)
        top_increasing = pos_contribs[:5]

        # Top 5 features decreasing churn probability (negative SHAP, ascending)
        neg_contribs = sorted([c for c in contributions if c.shap_value < 0], key=lambda x: x.shap_value)
        top_decreasing = neg_contribs[:5]

        # All contributions sorted by absolute magnitude
        all_sorted = sorted(contributions, key=lambda x: abs(x.shap_value), reverse=True)

        base_val = self.base_value if self.base_value is not None else 0.50

        # Executive summary narrative
        inc_text = ", ".join([f"{c.feature} (+{c.shap_value * 100:.1f}%)" for c in top_increasing[:3]]) if top_increasing else "None"
        dec_text = ", ".join([f"{c.feature} ({c.shap_value * 100:.1f}%)" for c in top_decreasing[:3]]) if top_decreasing else "None"
        summary = (
            f"Customer predicted churn probability is {churn_prob * 100:.1f}% (model baseline: {base_val * 100:.1f}%). "
            f"Primary factors driving churn risk up: {inc_text}. "
            f"Primary protective factors anchoring retention: {dec_text}."
        )

        return LocalExplanation(
            base_value=round(base_val, 4),
            predicted_probability=round(churn_prob, 4),
            top_churn_drivers=top_increasing,
            top_retention_factors=top_decreasing,
            all_contributions=all_sorted,
            summary=summary,
        )

    def predict(self, customer: CustomerInput) -> PredictionResponse:
        """Execute end-to-end churn prediction on a validated customer input with local SHAP explanation.

        Args:
            customer: Validated Pydantic CustomerInput model.

        Returns:
            PredictionResponse: Complete prediction result with probabilities, risk tiers, and local SHAP.

        Raises:
            HTTPException: 503 Service Unavailable if model is not loaded.
        """
        if not self.is_loaded or self.pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model service unavailable: {self.load_error or 'Pipeline not initialized'}",
            )

        # Convert validated customer data into an aligned single-row pandas DataFrame
        customer_dict = customer.model_dump()
        df_row = pd.DataFrame([customer_dict])[ORDERED_FEATURE_COLUMNS]

        # Predict churn probability (Class 1 = Churn)
        probabilities = self.pipeline.predict_proba(df_row)
        churn_prob = float(probabilities[0, 1])

        # Apply documented audit classification threshold (do not assume arbitrary default)
        threshold = CLASSIFICATION_THRESHOLD
        churn_pred = int(churn_prob >= threshold)
        churn_label = "Churn" if churn_pred == 1 else "No Churn"

        risk_tier = self.determine_risk_tier(churn_prob)
        recommendation = self.generate_recommendation(churn_prob, risk_tier, customer)

        # Generate genuine local SHAP explanation (gracefully handled if explainer unavailable)
        local_expl = None
        if self.explainer is not None:
            try:
                local_expl = self.explain_customer(df_row, churn_prob, customer)
            except Exception as err:
                logger.warning("Local SHAP explanation generation encountered an issue: %s", err)

        return PredictionResponse(
            churn_prediction=churn_pred,
            churn_label=churn_label,
            churn_probability=round(churn_prob, 4),
            risk_tier=risk_tier,
            classification_threshold=threshold,
            retention_recommendation=recommendation,
            explanation=local_expl,
        )

    def explain(self, customer: CustomerInput) -> ExplanationResponse:
        """Dedicated explanation method returning structured local SHAP diagnostics."""
        pred = self.predict(customer)
        if pred.explanation is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SHAP explanation could not be calculated for this customer.",
            )
        return ExplanationResponse(
            churn_prediction=pred.churn_prediction,
            churn_label=pred.churn_label,
            churn_probability=pred.churn_probability,
            risk_tier=pred.risk_tier,
            classification_threshold=pred.classification_threshold,
            retention_recommendation=pred.retention_recommendation,
            explanation=pred.explanation,
        )

    def get_benchmark_data(self) -> Dict[str, Any]:
        """Return the verified model benchmark comparison report from disk."""
        if AUDIT_REPORT_PATH.exists():
            try:
                with open(AUDIT_REPORT_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Could not read audit report from %s: %s", AUDIT_REPORT_PATH, e)
        return {
            "status": "unavailable",
            "message": f"Audit report not found at {AUDIT_REPORT_PATH}",
        }


# Global singleton instance
model_service = ModelService()
