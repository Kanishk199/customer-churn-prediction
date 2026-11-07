"""Unit and integration tests for local SHAP explainability."""

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from api.services import model_service, ORDERED_FEATURE_COLUMNS
from api.schemas import CustomerInput


@pytest.fixture(scope="module")
def client():
    """Create a TestClient instance managing app lifespan."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def high_risk_customer():
    """Sample customer representing a high-churn hazard."""
    return {
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


@pytest.fixture
def low_risk_customer():
    """Sample customer representing a loyal long-term subscriber."""
    return {
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


class TestShapCalculationsAndAlignment:
    """Tests verifying TreeExplainer initialization, efficiency property, and feature alignment."""

    def test_tree_explainer_initialization(self, client):
        """Verify model_service correctly initializes the TreeExplainer and extracts feature names."""
        assert model_service.is_loaded is True
        assert model_service.explainer is not None
        assert model_service.feature_names is not None
        assert len(model_service.feature_names) == 46
        assert model_service.base_value is not None
        assert 0.40 <= model_service.base_value <= 0.60

    def test_shap_efficiency_property(self, high_risk_customer):
        """Verify the SHAP efficiency property: base_value + sum(shap_values) == predict_proba."""
        customer = CustomerInput(**high_risk_customer)
        df_row = pd.DataFrame([customer.model_dump()])[ORDERED_FEATURE_COLUMNS]

        # Model predicted probability
        prob = float(model_service.pipeline.predict_proba(df_row)[0, 1])

        # Compute SHAP explanation
        explanation = model_service.explain_customer(df_row, prob, customer)

        # Sum of all local SHAP values
        shap_sum = sum(c.shap_value for c in explanation.all_contributions)
        reconstructed_prob = explanation.base_value + shap_sum

        # Efficiency property should hold within rounding tolerance
        assert abs(reconstructed_prob - prob) < 1e-3, (
            f"Efficiency mismatch: base ({explanation.base_value}) + sum ({shap_sum}) = "
            f"{reconstructed_prob} != model prob {prob}"
        )

    def test_feature_names_alignment(self, high_risk_customer):
        """Verify that SHAP contribution feature names strictly match preprocessor output names."""
        customer = CustomerInput(**high_risk_customer)
        df_row = pd.DataFrame([customer.model_dump()])[ORDERED_FEATURE_COLUMNS]

        preprocessor = model_service.pipeline.named_steps["preprocessor"]
        expected_names = list(preprocessor.get_feature_names_out())

        explanation = model_service.explain_customer(df_row, 0.90, customer)
        contribution_features = [c.feature for c in explanation.all_contributions]

        assert set(contribution_features) == set(expected_names)
        assert len(contribution_features) == len(expected_names) == 46

    def test_tree_explainer_output_space_is_probability(self):
        """Verify TreeExplainer outputs probability-space SHAP values directly matching predict_proba."""
        classifier = model_service.pipeline.named_steps["classifier"]
        explainer = model_service.explainer
        
        # In scikit-learn Random Forest, leaf nodes represent class probabilities.
        # TreeExplainer's 'raw' model output for RF directly computes probability attributions.
        assert explainer.model_output == "raw"
        
        # Base value represents the prior probability of churn in [0, 1]
        base_val = explainer.expected_value[1]
        assert 0.0 <= base_val <= 1.0, f"Expected probability base value in [0, 1], got {base_val}"
        assert abs(base_val - 0.500327) < 1e-4

    def test_positive_class_correspondence(self):
        """Verify that displayed probability and SHAP values correspond strictly to the positive class (Churn=1)."""
        classifier = model_service.pipeline.named_steps["classifier"]
        
        # Positive class corresponds to index 1 (class label 1, mapped from 'Yes')
        assert classifier.classes_[1] == 1
        assert classifier.classes_[0] == 0
        
        # Verify model_service uses column index 1 for churn probability
        sample_df = pd.DataFrame([{
            "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 95.0
        }])[ORDERED_FEATURE_COLUMNS]
        
        proba = model_service.pipeline.predict_proba(sample_df)[0]
        assert proba[1] > 0.50  # High-risk profile must have high positive churn probability

    def test_unrounded_mathematical_reconstruction_machine_precision(self, high_risk_customer, low_risk_customer):
        """Verify unrounded base_value + sum(shap_values) == predict_proba to machine precision (< 1e-12)."""
        preprocessor = model_service.pipeline.named_steps["preprocessor"]
        classifier = model_service.pipeline.named_steps["classifier"]
        explainer = model_service.explainer
        base_val = explainer.expected_value[1]
        
        for cust_dict in [high_risk_customer, low_risk_customer]:
            df_row = pd.DataFrame([cust_dict])[ORDERED_FEATURE_COLUMNS]
            true_proba = model_service.pipeline.predict_proba(df_row)[0, 1]
            
            X_trans = preprocessor.transform(df_row)
            sv = explainer.shap_values(X_trans)
            
            # sv shape is (1, 46, 2)
            churn_shap = sv[0, :, 1]
            reconstructed = base_val + np.sum(churn_shap)
            
            abs_diff = abs(true_proba - reconstructed)
            assert abs_diff < 1e-12, (
                f"Machine precision violation: true_proba={true_proba}, "
                f"reconstructed={reconstructed}, diff={abs_diff}"
            )

    def test_top_five_increasing_and_decreasing_features(self, high_risk_customer):
        """Verify top 5 increasing have positive SHAP and top 5 decreasing have negative SHAP."""
        customer = CustomerInput(**high_risk_customer)
        df_row = pd.DataFrame([customer.model_dump()])[ORDERED_FEATURE_COLUMNS]
        prob = float(model_service.pipeline.predict_proba(df_row)[0, 1])

        explanation = model_service.explain_customer(df_row, prob, customer)

        # Verify increasing drivers (up to 5)
        assert len(explanation.top_churn_drivers) <= 5
        assert len(explanation.top_churn_drivers) > 0
        for driver in explanation.top_churn_drivers:
            assert driver.shap_value > 0
            assert driver.impact == "increases_churn"
            assert driver.description is not None

        # Verify increasing drivers are sorted descending
        inc_values = [d.shap_value for d in explanation.top_churn_drivers]
        assert inc_values == sorted(inc_values, reverse=True)

        # Verify decreasing drivers (up to 5)
        assert len(explanation.top_retention_factors) <= 5
        assert len(explanation.top_retention_factors) > 0
        for factor in explanation.top_retention_factors:
            assert factor.shap_value < 0
            assert factor.impact == "decreases_churn"
            assert factor.description is not None

        # Verify decreasing factors are sorted ascending (most negative first)
        dec_values = [d.shap_value for d in explanation.top_retention_factors]
        assert dec_values == sorted(dec_values)


class TestApiExplanationEndpoints:
    """Tests verifying FastAPI explainability endpoints and prediction correspondence."""

    def test_predict_endpoint_returns_explanation(self, client, high_risk_customer):
        """Verify POST /predict response includes populated local SHAP explanation."""
        resp = client.post("/predict", json=high_risk_customer)
        assert resp.status_code == 200
        data = resp.json()

        assert "explanation" in data
        assert data["explanation"] is not None

        expl = data["explanation"]
        assert "base_value" in expl
        assert "predicted_probability" in expl
        assert "top_churn_drivers" in expl
        assert "top_retention_factors" in expl
        assert "summary" in expl

        # Prediction correspondence check
        assert round(data["churn_probability"], 4) == round(expl["predicted_probability"], 4)
        assert len(expl["top_churn_drivers"]) <= 5
        assert len(expl["top_retention_factors"]) <= 5
        assert len(expl["summary"]) > 20

    def test_dedicated_explain_endpoint(self, client, low_risk_customer):
        """Verify POST /explain returns complete ExplanationResponse matching low-risk customer."""
        resp = client.post("/explain", json=low_risk_customer)
        assert resp.status_code == 200
        data = resp.json()

        assert data["churn_prediction"] == 0
        assert data["churn_label"] == "No Churn"
        assert data["risk_tier"] == "Low Risk"

        expl = data["explanation"]
        assert abs(expl["base_value"] - model_service.base_value) < 1e-4
        assert expl["predicted_probability"] == data["churn_probability"]
        assert len(expl["all_contributions"]) == 46

        # For a loyal customer, top retention factors should include contract or tenure
        retention_features = [f["feature"] for f in expl["top_retention_factors"]]
        assert any("Contract" in f or "tenure" in f for f in retention_features)

    def test_explanation_matches_exact_prediction(self, client, high_risk_customer):
        """Verify calling /predict and /explain on the same customer yields identical predictions and SHAP."""
        pred_resp = client.post("/predict", json=high_risk_customer).json()
        expl_resp = client.post("/explain", json=high_risk_customer).json()

        assert pred_resp["churn_probability"] == expl_resp["churn_probability"]
        assert pred_resp["churn_prediction"] == expl_resp["churn_prediction"]
        assert pred_resp["risk_tier"] == expl_resp["risk_tier"]
        assert pred_resp["explanation"]["base_value"] == expl_resp["explanation"]["base_value"]
        assert pred_resp["explanation"]["predicted_probability"] == expl_resp["explanation"]["predicted_probability"]

        # Assert top drivers match identically
        pred_top_inc = [d["feature"] for d in pred_resp["explanation"]["top_churn_drivers"]]
        expl_top_inc = [d["feature"] for d in expl_resp["explanation"]["top_churn_drivers"]]
        assert pred_top_inc == expl_top_inc

    def test_shap_handles_edge_cases_cleanly(self, client, high_risk_customer):
        """Verify SHAP explanation handles tenure == 0 and omitted TotalCharges without breaking."""
        edge_customer = high_risk_customer.copy()
        edge_customer["tenure"] = 0
        del edge_customer["TotalCharges"]

        resp = client.post("/explain", json=edge_customer)
        assert resp.status_code == 200
        data = resp.json()
        assert data["explanation"] is not None
        assert np.isfinite(data["explanation"]["predicted_probability"])
        for c in data["explanation"]["top_churn_drivers"]:
            assert np.isfinite(c["shap_value"])
            assert np.isfinite(c["feature_value"])
