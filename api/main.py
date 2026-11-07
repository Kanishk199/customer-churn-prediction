"""FastAPI application entrypoint for Telco Customer Churn Prediction."""

from contextlib import asynccontextmanager
import logging
import time
from typing import Dict, Any
from fastapi import FastAPI, status, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

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


TAGS_METADATA = [
    {
        "name": "General",
        "description": "Core service metadata, directory, and status endpoints.",
    },
    {
        "name": "Diagnostics",
        "description": "System health probes, model readiness, and verified benchmark metrics.",
    },
    {
        "name": "Inference",
        "description": "Real-time customer churn risk evaluation and retention playbooks.",
    },
    {
        "name": "Explainability",
        "description": "Local feature attribution engine powered by SHAP TreeExplainer.",
    },
]


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
    openapi_tags=TAGS_METADATA,
    contact={
        "name": "Kanishk",
        "url": "https://github.com/Kanishk199/customer-churn-prediction",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Enable Cross-Origin Resource Sharing (CORS) for decoupled dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Inject inference and processing latency telemetry header."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
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
        benchmark_url="/benchmark",
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


@app.get(
    "/benchmark",
    tags=["Diagnostics"],
    summary="Verified model benchmark comparison and evaluation metrics",
)
async def benchmark() -> Dict[str, Any]:
    """Retrieve verified evaluation metrics, hyperparameters, and comparison results from the model audit."""
    return model_service.get_benchmark_data()


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
