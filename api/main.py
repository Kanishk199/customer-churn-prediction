"""FastAPI application entrypoint for Telco Customer Churn Prediction."""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from api.config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    CLASSIFICATION_THRESHOLD,
    LOW_RISK_UPPER_BOUND,
    MEDIUM_RISK_UPPER_BOUND,
)
from api.schemas import (
    CustomerInput,
    PredictionResponse,
    HealthResponse,
    RootResponse,
    ExplanationResponse,
)
from api.services import model_service, ORDERED_FEATURE_COLUMNS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to initialize the model on startup."""
    logger.info("Starting up Telco Churn Prediction API...")
    model_service.load_model()
    if model_service.is_loaded:
        logger.info("Model pipeline loaded and ready for inference.")
    else:
        logger.error("Failed to load model pipeline on startup: %s", model_service.load_error)
    yield
    logger.info("Shutting down Telco Churn Prediction API.")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    """Provide formatted, readable error messages for validation failures."""
    errors = []
    for error in exc.errors():
        field = " -> ".join([str(loc) for loc in error.get("loc", []) if loc != "body"])
        errors.append({
            "field": field,
            "message": error.get("msg"),
            "type": error.get("type"),
        })
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": "One or more input fields failed validation checks.",
            "field_errors": errors,
        },
    )


@app.get(
    "/",
    response_model=RootResponse,
    tags=["General"],
    summary="Root welcome and API directory",
)
async def root() -> RootResponse:
    """Return API metadata, version, and links to documentation and endpoints."""
    return RootResponse(
        message="Welcome to the Telco Customer Churn Prediction API",
        version=API_VERSION,
        status="online",
        docs_url="/docs",
        health_url="/health",
        predict_url="/predict",
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Diagnostics"],
    summary="Service health and model readiness check",
)
async def health() -> HealthResponse:
    """Diagnostic health check verifying model pipeline status and threshold configurations."""
    is_healthy = model_service.is_loaded
    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        model_loaded=model_service.is_loaded,
        model_type="Random Forest Classifier" if is_healthy else "Unavailable",
        version=API_VERSION,
        classification_threshold=CLASSIFICATION_THRESHOLD,
        low_risk_upper_bound=LOW_RISK_UPPER_BOUND,
        medium_risk_upper_bound=MEDIUM_RISK_UPPER_BOUND,
        features_expected=len(ORDERED_FEATURE_COLUMNS),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Predict customer churn probability and risk tier",
)
async def predict_churn(customer: CustomerInput) -> PredictionResponse:
    """Accept validated customer features, run pipeline inference, and return predictions.

    - **Binary Churn Decision:** Evaluated using the calibrated threshold from the model audit.
    - **Probability:** Raw output probability from the Random Forest ensemble.
    - **Risk Stratification:** Categorized into Low, Medium, or High Risk tiers.
    - **Prescriptive Guidance:** Suggests targeted retention actions.
    - **Local Explainability:** Contains genuine local SHAP feature contributions.
    """
    return model_service.predict(customer)


@app.post(
    "/explain",
    response_model=ExplanationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Explainability"],
    summary="Generate local SHAP feature attributions for a customer prediction",
)
async def explain_customer(customer: CustomerInput) -> ExplanationResponse:
    """Calculate genuine local SHAP feature contributions using TreeExplainer.

    - **Base Value:** Prior expected probability of the ensemble.
    - **Top 5 Churn Drivers:** Features increasing churn probability (+ contribution).
    - **Top 5 Retention Anchors:** Features decreasing churn probability (- contribution).
    - **Aligned Feature Names:** Preserves preprocessor one-hot and scaled feature names.
    """
    return model_service.explain(customer)
