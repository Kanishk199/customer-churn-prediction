"""Automated tests for FastAPI Customer Churn Prediction Backend."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services import model_service
from api.config import CLASSIFICATION_THRESHOLD


@pytest.fixture(scope="module")
def client():
    """Create a TestClient instance managing app lifespan."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_high_risk_customer():
    """Customer features representing typical high-risk profile (new, month-to-month, fiber optic, electronic check)."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": 95.50,
    }


@pytest.fixture
def sample_low_risk_customer():
    """Customer features representing typical low-risk loyal profile (long tenure, 2-year contract, bundled services)."""
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 60.00,
        "TotalCharges": 3900.00,
    }


class TestApiEndpoints:
    """Tests for core API routes, schemas, and health reporting."""

    def test_root_endpoint(self, client):
        """Verify GET / returns API metadata and directory links."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert data["version"] == "1.0.0"
        assert data["docs_url"] == "/docs"
        assert data["health_url"] == "/health"
        assert data["predict_url"] == "/predict"

    def test_health_endpoint_healthy(self, client):
        """Verify GET /health returns 200 and confirms model is loaded."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_type"] == "Random Forest Classifier"
        assert data["classification_threshold"] == CLASSIFICATION_THRESHOLD
        assert data["features_expected"] == 19

    def test_predict_high_risk_customer(self, client, sample_high_risk_customer):
        """Verify POST /predict correctly flags a high-risk churner."""
        response = client.post("/predict", json=sample_high_risk_customer)
        assert response.status_code == 200
        data = response.json()

        assert "churn_prediction" in data
        assert "churn_label" in data
        assert "churn_probability" in data
        assert "risk_tier" in data
        assert "classification_threshold" in data
        assert "retention_recommendation" in data

        # Check high-risk characteristics
        assert data["churn_prediction"] == 1
        assert data["churn_label"] == "Churn"
        assert data["churn_probability"] >= CLASSIFICATION_THRESHOLD
        assert data["risk_tier"] == "High Risk"
        assert "retention" in data["retention_recommendation"].lower()

    def test_predict_low_risk_customer(self, client, sample_low_risk_customer):
        """Verify POST /predict correctly flags a retained customer as Low Risk."""
        response = client.post("/predict", json=sample_low_risk_customer)
        assert response.status_code == 200
        data = response.json()

        assert data["churn_prediction"] == 0
        assert data["churn_label"] == "No Churn"
        assert data["churn_probability"] < 0.30
        assert data["risk_tier"] == "Low Risk"
        assert data["classification_threshold"] == CLASSIFICATION_THRESHOLD

    def test_predict_missing_total_charges_auto_imputed(self, client, sample_high_risk_customer):
        """Verify TotalCharges defaults properly if omitted (e.g. new user)."""
        payload = sample_high_risk_customer.copy()
        payload["tenure"] = 0
        del payload["TotalCharges"]

        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_tier"] in ["High Risk", "Medium Risk"]

    def test_predict_invalid_enum_rejected(self, client, sample_high_risk_customer):
        """Verify POST /predict rejects unrecognized categorical levels with HTTP 422."""
        payload = sample_high_risk_customer.copy()
        payload["Contract"] = "Centennial-Contract"  # Invalid enum value

        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert data["error"] == "Validation Error"
        assert any("Contract" in str(err["field"]) for err in data["field_errors"])

    def test_predict_invalid_numeric_range_rejected(self, client, sample_high_risk_customer):
        """Verify POST /predict rejects negative tenure or charges with HTTP 422."""
        payload = sample_high_risk_customer.copy()
        payload["tenure"] = -5  # Negative tenure

        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert any("tenure" in str(err["field"]) for err in data["field_errors"])

    def test_predict_missing_required_fields_rejected(self, client):
        """Verify POST /predict rejects empty or incomplete payload with HTTP 422."""
        response = client.post("/predict", json={"gender": "Female"})
        assert response.status_code == 422
        data = response.json()
        assert len(data["field_errors"]) > 0

    def test_service_unavailable_when_model_unloaded(self, client, sample_low_risk_customer):
        """Verify POST /predict returns 503 if model pipeline fails to load."""
        original_pipeline = model_service.pipeline
        original_status = model_service.is_loaded

        try:
            # Simulate unloaded model
            model_service.pipeline = None
            model_service.is_loaded = False
            model_service.load_error = "Simulated disk failure"

            # Check health endpoint returns degraded
            health_resp = client.get("/health")
            assert health_resp.status_code == 200
            assert health_resp.json()["status"] == "degraded"
            assert health_resp.json()["model_loaded"] is False

            # Check predict endpoint returns 503
            predict_resp = client.post("/predict", json=sample_low_risk_customer)
            assert predict_resp.status_code == 503
            assert "Model service unavailable" in predict_resp.json()["detail"]
        finally:
            # Restore model service
            model_service.pipeline = original_pipeline
            model_service.is_loaded = original_status
            model_service.load_error = None
