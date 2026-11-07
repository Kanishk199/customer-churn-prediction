"""Pydantic schemas for request validation and response serialization."""

from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator


class CustomerInput(BaseModel):
    """Customer features schema expected by the trained churn pipeline."""

    gender: Literal["Female", "Male"] = Field(
        ...,
        description="Customer gender identity",
        examples=["Female"],
    )
    SeniorCitizen: int = Field(
        ...,
        ge=0,
        le=1,
        description="Whether customer is a senior citizen (1) or not (0)",
        examples=[0],
    )
    Partner: Literal["Yes", "No"] = Field(
        ...,
        description="Whether customer has a partner",
        examples=["Yes"],
    )
    Dependents: Literal["Yes", "No"] = Field(
        ...,
        description="Whether customer has dependents",
        examples=["No"],
    )
    tenure: int = Field(
        ...,
        ge=0,
        le=120,
        description="Number of months customer has remained subscribed with the company",
        examples=[1],
    )
    PhoneService: Literal["Yes", "No"] = Field(
        ...,
        description="Whether customer has a phone service",
        examples=["No"],
    )
    MultipleLines: Literal["No", "Yes", "No phone service"] = Field(
        ...,
        description="Whether customer has multiple telephone lines",
        examples=["No phone service"],
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ...,
        description="Customer's internet service provider technology",
        examples=["DSL"],
    )
    OnlineSecurity: Literal["No", "Yes", "No internet service"] = Field(
        ...,
        description="Whether customer has online security add-on",
        examples=["No"],
    )
    OnlineBackup: Literal["No", "Yes", "No internet service"] = Field(
        ...,
        description="Whether customer has online backup add-on",
        examples=["Yes"],
    )
    DeviceProtection: Literal["No", "Yes", "No internet service"] = Field(
        ...,
        description="Whether customer has device protection plan",
        examples=["No"],
    )
    TechSupport: Literal["No", "Yes", "No internet service"] = Field(
        ...,
        description="Whether customer has premium technical support",
        examples=["No"],
    )
    StreamingTV: Literal["No", "Yes", "No internet service"] = Field(
        ...,
        description="Whether customer streams third-party TV programs",
        examples=["No"],
    )
    StreamingMovies: Literal["No", "Yes", "No internet service"] = Field(
        ...,
        description="Whether customer streams third-party movies",
        examples=["No"],
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ...,
        description="Contract agreement term duration",
        examples=["Month-to-month"],
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ...,
        description="Whether customer has paperless electronic billing enabled",
        examples=["Yes"],
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(
        ...,
        description="Customer payment method",
        examples=["Electronic check"],
    )
    MonthlyCharges: float = Field(
        ...,
        ge=0.0,
        le=1000.0,
        description="Monthly recurring charges billed to the customer in USD",
        examples=[29.85],
    )
    TotalCharges: Optional[float] = Field(
        None,
        ge=0.0,
        description="Cumulative total amount charged to customer; defaults to 0.0 for new subscribers",
        examples=[29.85],
    )

    @model_validator(mode="after")
    def validate_total_charges(self) -> "CustomerInput":
        """Impute TotalCharges to 0.0 if missing or tenure == 0."""
        if self.TotalCharges is None:
            if self.tenure == 0:
                self.TotalCharges = 0.0
            else:
                self.TotalCharges = float(self.MonthlyCharges * self.tenure)
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }
    )


class FeatureContribution(BaseModel):
    """Individual feature contribution to customer churn prediction via local SHAP."""

    feature: str = Field(
        ...,
        description="Transformed feature identifier used in the pipeline",
        examples=["Contract_Month-to-month"],
    )
    feature_value: float = Field(
        ...,
        description="Customer transformed feature value",
        examples=[1.0],
    )
    shap_value: float = Field(
        ...,
        description="SHAP attribution value (contribution to positive churn probability)",
        examples=[0.0479],
    )
    impact: Literal["increases_churn", "decreases_churn"] = Field(
        ...,
        description="Direction of impact on churn probability",
        examples=["increases_churn"],
    )
    description: Optional[str] = Field(
        None,
        description="Human-readable explanation of this factor's influence",
        examples=["Customer is enrolled in a flexible month-to-month contract (+4.8% churn risk)."],
    )


class LocalExplanation(BaseModel):
    """Structured local SHAP explainability breakdown for a specific customer prediction."""

    base_value: float = Field(
        ...,
        description="Model expected base probability prior to observing customer features",
        examples=[0.5003],
    )
    predicted_probability: float = Field(
        ...,
        description="Final model predicted probability for this customer",
        examples=[0.9095],
    )
    top_churn_drivers: List[FeatureContribution] = Field(
        ...,
        description="Top 5 features pushing the customer toward churn (positive SHAP values)",
    )
    top_retention_factors: List[FeatureContribution] = Field(
        ...,
        description="Top 5 features anchoring customer retention (negative SHAP values)",
    )
    all_contributions: List[FeatureContribution] = Field(
        default_factory=list,
        description="Full sorted list of all active feature attributions",
    )
    summary: str = Field(
        ...,
        description="Executive summary of key drivers influencing this customer",
        examples=["Primary drivers increasing churn risk are Month-to-month contract and low tenure."],
    )


class PredictionResponse(BaseModel):
    """Schema for churn prediction output."""

    churn_prediction: int = Field(
        ...,
        description="Binary churn prediction: 1 for Churn, 0 for Retained",
        examples=[1],
    )
    churn_label: Literal["Churn", "No Churn"] = Field(
        ...,
        description="Human-readable prediction label",
        examples=["Churn"],
    )
    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model estimated probability of customer churning",
        examples=[0.7000],
    )
    risk_tier: Literal["Low Risk", "Medium Risk", "High Risk"] = Field(
        ...,
        description="Stratified customer risk category based on audit thresholds",
        examples=["High Risk"],
    )
    classification_threshold: float = Field(
        ...,
        description="Calibrated decision threshold applied from model audit",
        examples=[0.50],
    )
    retention_recommendation: str = Field(
        ...,
        description="Prescriptive business action based on risk profile and drivers",
        examples=["Immediate outreach: offer 1-year contract extension with bundled tech support."],
    )
    explanation: Optional[LocalExplanation] = Field(
        None,
        description="Genuine local SHAP explainability breakdown for this customer",
    )


class ExplanationResponse(BaseModel):
    """Schema for dedicated customer explanation endpoint."""

    churn_prediction: int = Field(..., description="Binary churn prediction (1 for Churn, 0 for Retained)")
    churn_label: Literal["Churn", "No Churn"] = Field(..., description="Human-readable prediction label")
    churn_probability: float = Field(..., description="Estimated churn probability")
    risk_tier: Literal["Low Risk", "Medium Risk", "High Risk"] = Field(..., description="Risk category")
    classification_threshold: float = Field(..., description="Decision threshold applied")
    retention_recommendation: str = Field(..., description="Prescriptive retention action")
    explanation: LocalExplanation = Field(..., description="Detailed local SHAP attribution results")


class HealthResponse(BaseModel):
    """Schema for API health status check."""

    status: Literal["healthy", "degraded"] = Field(..., examples=["healthy"])
    model_loaded: bool = Field(..., examples=[True])
    model_type: str = Field(..., examples=["Random Forest Classifier"])
    version: str = Field(..., examples=["1.0.0"])
    classification_threshold: float = Field(..., examples=[0.50])
    low_risk_upper_bound: float = Field(..., examples=[0.30])
    medium_risk_upper_bound: float = Field(..., examples=[0.60])
    features_expected: int = Field(..., examples=[19])


class RootResponse(BaseModel):
    """Schema for root endpoint response."""

    message: str = Field(..., examples=["Welcome to the Telco Customer Churn Prediction API"])
    version: str = Field(..., examples=["1.0.0"])
    status: str = Field(..., examples=["online"])
    docs_url: str = Field(..., examples=["/docs"])
    health_url: str = Field(..., examples=["/health"])
    predict_url: str = Field(..., examples=["/predict"])
