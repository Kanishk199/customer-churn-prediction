"""Verification script for containerized API prediction and SHAP attributions."""

import requests

loyal_customer = {
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

response = requests.post("http://localhost:8000/predict", json=loyal_customer)
print("Loyal Customer Status Code:", response.status_code)
data = response.json()

print(f"Prediction: {data['churn_label']} (Class {data['churn_prediction']})")
print(f"Probability: {data['churn_probability'] * 100:.2f}%")
print(f"Risk Tier: {data['risk_tier']}")
print(f"Recommendation: {data['retention_recommendation']}")

expl = data["explanation"]
print(f"SHAP Base: {expl['base_value'] * 100:.2f}%")
print("Top Retention Drivers:")
for d in expl["top_retention_factors"][:3]:
    print(f"  [-] {d['feature']}: {d['shap_value']:+.4f} -> {d['description']}")
