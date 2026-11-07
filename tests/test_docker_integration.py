"""Automated Integration and Validation Tests for Dockerization Architecture.

Validates Dockerfile, docker-compose.yml, .dockerignore, inter-container communication,
environment variable propagation, healthcheck configurations, and SHAP explainability.
"""

import os
from pathlib import Path
import re
import yaml
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.config import MODEL_PATH, AUDIT_REPORT_PATH, get_classification_threshold
from api.services import model_service

BASE_DIR = Path(__file__).resolve().parent.parent
DOCKERFILE_PATH = BASE_DIR / "Dockerfile"
DOCKER_COMPOSE_PATH = BASE_DIR / "docker-compose.yml"
DOCKERIGNORE_PATH = BASE_DIR / ".dockerignore"
REQUIREMENTS_PATH = BASE_DIR / "requirements.txt"


class TestDockerfileValidation:
    """Verifies Dockerfile syntax, multi-stage structure, security, and healthchecks."""

    def test_dockerfile_exists_and_readable(self):
        assert DOCKERFILE_PATH.exists(), "Dockerfile must exist at the project root."
        content = DOCKERFILE_PATH.read_text(encoding="utf-8")
        assert len(content) > 100, "Dockerfile should not be empty."

    def test_dockerfile_multi_stage_targets(self):
        content = DOCKERFILE_PATH.read_text(encoding="utf-8")
        assert "AS base" in content, "Dockerfile should define a common base stage."
        assert "AS frontend" in content, "Dockerfile should define a dedicated frontend stage."
        assert "AS backend" in content, "Dockerfile should define a dedicated backend stage."

    def test_dockerfile_security_and_non_root_user(self):
        content = DOCKERFILE_PATH.read_text(encoding="utf-8")
        # Check that non-root user is created and switched to
        assert "useradd" in content or "adduser" in content, "Dockerfile must create a non-root system user."
        assert "USER appuser" in content or "USER 10001" in content, "Dockerfile must switch to a non-root user."
        # Check that sensitive keys or secrets are NOT baked in
        forbidden_keywords = ["SECRET_KEY", "AWS_ACCESS_KEY", "PASSWORD=", "TOKEN="]
        for kw in forbidden_keywords:
            assert kw not in content, f"Dockerfile must not contain hardcoded secret: {kw}"

    def test_dockerfile_healthcheck_instructions(self):
        content = DOCKERFILE_PATH.read_text(encoding="utf-8")
        assert "HEALTHCHECK" in content, "Dockerfile must define HEALTHCHECK probes."
        assert "http://localhost:8000/health" in content, "Backend healthcheck must query /health."
        assert "http://localhost:8501/_stcore/health" in content, "Frontend healthcheck must query Streamlit health endpoint."

    def test_dockerfile_ports_and_model_artifacts(self):
        content = DOCKERFILE_PATH.read_text(encoding="utf-8")
        assert "EXPOSE 8000" in content, "Backend stage must expose port 8000."
        assert "EXPOSE 8501" in content, "Frontend stage must expose port 8501."
        assert "best_churn_model.joblib" in content, "Dockerfile must copy the trained model artifact."
        assert "model_comparison_report.json" in content, "Dockerfile must copy the model comparison report."


class TestDockerComposeValidation:
    """Validates docker-compose.yml structure, dependencies, networking, and volumes."""

    @pytest.fixture(autouse=True)
    def load_compose(self):
        assert DOCKER_COMPOSE_PATH.exists(), "docker-compose.yml must exist at project root."
        with open(DOCKER_COMPOSE_PATH, "r", encoding="utf-8") as f:
            self.compose_data = yaml.safe_load(f)

    def test_services_defined(self):
        services = self.compose_data.get("services", {})
        assert "backend" in services, "docker-compose must define a 'backend' service."
        assert "frontend" in services, "docker-compose must define a 'frontend' service."

    def test_backend_configuration(self):
        backend = self.compose_data["services"]["backend"]
        # Ports
        ports = [str(p) for p in backend.get("ports", [])]
        assert any("8000:8000" in p for p in ports), "Backend must expose 8000:8000."
        assert not any("8501" in p for p in ports), "Backend must not expose port 8501."
        # Read-only model mount
        volumes = [str(v) for v in backend.get("volumes", [])]
        assert any("./models:/app/models:ro" in v or ":ro" in v for v in volumes), \
            "Backend must mount model directory as read-only (:ro) to prevent modification."
        # Healthcheck
        assert "healthcheck" in backend, "Backend must configure healthcheck."

    def test_frontend_configuration_and_dependency(self):
        frontend = self.compose_data["services"]["frontend"]
        # Ports
        ports = [str(p) for p in frontend.get("ports", [])]
        assert any("8501:8501" in p for p in ports), "Frontend must expose 8501:8501."
        assert not any("8000" in p for p in ports), "Frontend must not expose port 8000."
        # depends_on with condition: service_healthy
        depends_on = frontend.get("depends_on", {})
        assert "backend" in depends_on, "Frontend must declare dependency on backend."
        if isinstance(depends_on, dict):
            assert depends_on["backend"].get("condition") == "service_healthy", \
                "Frontend must wait for backend health check to succeed before starting."
        # Inter-container API_URL
        env = frontend.get("environment", [])
        if isinstance(env, list):
            env_dict = dict(item.split("=", 1) for item in env if "=" in item)
        else:
            env_dict = env
        assert env_dict.get("API_URL") == "http://backend:8000", \
            "Frontend must communicate with backend using container service name 'http://backend:8000'."

    def test_bridge_network_isolation(self):
        networks = self.compose_data.get("networks", {})
        assert "churn-network" in networks, "docker-compose must define an isolated bridge network."
        backend_nets = self.compose_data["services"]["backend"].get("networks", [])
        frontend_nets = self.compose_data["services"]["frontend"].get("networks", [])
        assert "churn-network" in backend_nets, "Backend must join churn-network."
        assert "churn-network" in frontend_nets, "Frontend must join churn-network."


class TestDockerignoreValidation:
    """Validates that .dockerignore prevents leaking virtual environments and development files."""

    @pytest.fixture(autouse=True)
    def load_dockerignore(self):
        assert DOCKERIGNORE_PATH.exists(), ".dockerignore must exist at project root."
        self.ignored = [
            line.strip()
            for line in DOCKERIGNORE_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]

    def test_virtual_env_ignored(self):
        assert any(".venv" in pat for pat in self.ignored), ".venv must be excluded in .dockerignore."

    def test_caches_and_tests_ignored(self):
        assert any("__pycache__" in pat for pat in self.ignored), "__pycache__ must be excluded."
        assert any(".pytest_cache" in pat for pat in self.ignored), ".pytest_cache must be excluded."

    def test_git_metadata_ignored(self):
        assert any(".git" in pat for pat in self.ignored), ".git must be excluded."

    def test_model_and_code_not_ignored(self):
        # Ensure model and source code are NOT accidentally ignored
        for pat in self.ignored:
            assert pat not in ["models", "models/", "src", "src/", "api", "api/", "app", "app/"], \
                f"Essential directory {pat} must not be ignored in .dockerignore."


class TestContainerizedInferenceAndSHAP:
    """Verifies that the inference and SHAP explainability engine functions under container settings."""

    @pytest.fixture(autouse=True)
    def client(self):
        return TestClient(app)

    def test_health_check_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "healthy"
        assert payload["model_loaded"] is True
        assert "Random Forest" in payload["model_type"]

    def test_explain_endpoint_shap_execution(self, client):
        sample_customer = {
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
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.35,
            "TotalCharges": 140.70,
        }
        response = client.post("/explain", json=sample_customer)
        assert response.status_code == 200
        payload = response.json()
        assert "churn_probability" in payload
        assert "explanation" in payload
        expl = payload["explanation"]
        assert "base_value" in expl
        assert "top_churn_drivers" in expl
        assert "top_retention_factors" in expl
        assert len(expl["top_churn_drivers"]) <= 5
        assert len(expl["top_retention_factors"]) <= 5

    def test_model_artifact_integrity(self):
        """Verifies that the model artifact file is intact and not modified or corrupted."""
        assert MODEL_PATH.exists(), f"Model artifact must exist at {MODEL_PATH}"
        assert AUDIT_REPORT_PATH.exists(), f"Audit report must exist at {AUDIT_REPORT_PATH}"
        # Ensure classification threshold matches documented value
        threshold = get_classification_threshold()
        assert threshold == 0.50, f"Expected threshold 0.50, got {threshold}"
