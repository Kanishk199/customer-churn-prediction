"""Verification script for containerized API prediction and SHAP attributions."""

import json
import requests

customer = {
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
    "MonthlyCharges": 95.50,
    "TotalCharges": 191.00,
}

response = requests.post("http://localhost:8000/predict", json=customer)
print("Status Code:", response.status_code)
data = response.json()

print(f"Prediction: {data['churn_label']} (Class {data['churn_prediction']})")
print(f"Probability: {data['churn_probability'] * 100:.2f}%")
print(f"Risk Tier: {data['risk_tier']}")
print(f"Decision Threshold: {data['classification_threshold']}")
print(f"Recommendation: {data['retention_recommendation']}")

expl = data["explanation"]
print(f"\nSHAP Base Probability: {expl['base_value'] * 100:.2f}%")
print("Top 5 Factors Increasing Churn Risk (+):")
for d in expl["top_churn_drivers"]:
    print(f"  [+] {d['feature']}: {d['shap_value']:+.4f} -> {d['description']}")

print("\nTop Factors Decreasing Churn Risk (-):")
for d in expl["top_retention_factors"]:
    print(f"  [-] {d['feature']}: {d['shap_value']:+.4f} -> {d['description']}")

print(f"\nExecutive Summary: {expl['summary']}")
