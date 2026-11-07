"""End-to-end integration tests validating Streamlit app logic against live FastAPI backend."""

import pytest
import requests

API_URL = "http://127.0.0.1:8000"


class TestFrontendBackendIntegration:
    """Verifies that Streamlit payload structures and communications work with the live FastAPI backend."""

    def test_backend_health_probe(self):
        """Verify the Streamlit health probe successfully queries GET /health."""
        resp = requests.get(f"{API_URL}/health", timeout=3.0)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_type"] == "Random Forest Classifier"

    def test_high_risk_preset_integration(self):
        """Verify the Streamlit 'At-Risk Customer' preset payload yields High Risk from backend."""
        high_risk_payload = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
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
            "MonthlyCharges": 92.50,
            "TotalCharges": 185.00,
        }

        resp = requests.post(f"{API_URL}/predict", json=high_risk_payload, timeout=5.0)
        assert resp.status_code == 200
        data = resp.json()

        assert data["churn_prediction"] == 1
        assert data["churn_label"] == "Churn"
        assert data["churn_probability"] >= 0.60
        assert data["risk_tier"] == "High Risk"
        assert len(data["retention_recommendation"]) > 10

    def test_loyal_subscriber_preset_integration(self):
        """Verify the Streamlit 'Loyal Subscriber' preset payload yields Low Risk from backend."""
        loyal_payload = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 62,
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
            "MonthlyCharges": 64.00,
            "TotalCharges": 3968.00,
        }

        resp = requests.post(f"{API_URL}/predict", json=loyal_payload, timeout=5.0)
        assert resp.status_code == 200
        data = resp.json()

        assert data["churn_prediction"] == 0
        assert data["churn_label"] == "No Churn"
        assert data["churn_probability"] < 0.30
        assert data["risk_tier"] == "Low Risk"

    def test_moderate_risk_preset_integration(self):
        """Verify the Streamlit 'Moderate Risk' preset payload yields valid response."""
        moderate_payload = {
            "gender": "Female",
            "SeniorCitizen": 1,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 18,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 84.50,
            "TotalCharges": 1521.00,
        }

        resp = requests.post(f"{API_URL}/predict", json=moderate_payload, timeout=5.0)
        assert resp.status_code == 200
        data = resp.json()

        assert "churn_probability" in data
        assert data["risk_tier"] in ["Medium Risk", "High Risk", "Low Risk"]
        assert data["classification_threshold"] == 0.50

    def test_streamlit_app_code_does_not_load_models_or_sklearn(self):
        """Audit safeguard: verify Streamlit file does not import joblib or load models directly."""
        app_file = "app/streamlit_app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "joblib" not in content, "Streamlit app must not import or use joblib directly"
        assert "fit(" not in content, "Streamlit app must never call fit() on models"
        assert "sklearn" not in content, "Streamlit app must not import scikit-learn"
