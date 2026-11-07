"""Professional Streamlit Dashboard for Telco Customer Churn Prediction.

Connects strictly via HTTP REST to the FastAPI backend service.
Does NOT load or train any ML models within Streamlit.
"""

import os
import requests
import streamlit as st
import matplotlib.pyplot as plt

# Configure page metadata and layout
st.set_page_config(
    page_title="Telco Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom SaaS-styled CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .badge-high {
        color: #DC2626;
        background-color: #FEE2E2;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .badge-medium {
        color: #D97706;
        background-color: #FEF3C7;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .badge-low {
        color: #16A34A;
        background-color: #DCFCE7;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .playbook-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Backend Connection & Persona Templates
st.sidebar.title("⚙️ Backend Configuration")
api_base_url = st.sidebar.text_input(
    "FastAPI Base URL",
    value=os.getenv("API_URL", "http://127.0.0.1:8000"),
    help="Target host and port for the running FastAPI service.",
)

# Health Check Probe
backend_healthy = False
health_data = {}
try:
    health_resp = requests.get(f"{api_base_url}/health", timeout=2.5)
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        backend_healthy = health_data.get("status") == "healthy"
except Exception:
    backend_healthy = False

if backend_healthy:
    st.sidebar.success(f"🟢 Backend Online (v{health_data.get('version', '1.0')})")
    st.sidebar.caption(
        f"**Model:** {health_data.get('model_type')}\n\n"
        f"**Audit Decision Threshold:** `{health_data.get('classification_threshold', 0.50):.2f}`\n\n"
        f"**Risk Cutoffs:** Low < `{health_data.get('low_risk_upper_bound')}` | Med < `{health_data.get('medium_risk_upper_bound')}`"
    )
else:
    st.sidebar.error("🔴 Backend Offline or Unreachable")
    st.sidebar.warning(
        "Make sure FastAPI is running locally:\n"
        "`uvicorn api.main:app --host 127.0.0.1 --port 8000`"
    )

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Customer Profile Presets")
preset_choice = st.sidebar.selectbox(
    "Load Pre-configured Persona",
    [
        "Custom Input",
        "🚨 At-Risk Customer (Month-to-month, Fiber, High Bills)",
        "🛡️ Loyal Subscriber (Two-year, DSL, Full Protection)",
        "⚖️ Moderate Risk Customer (1-Year, Fiber, Middle Tenure)",
    ],
)

# Preset feature initializers
default_values = {
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

if preset_choice.startswith("🛡️ Loyal"):
    default_values.update({
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
    })
elif preset_choice.startswith("⚖️ Moderate"):
    default_values.update({
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
    })

# Main Dashboard View
st.markdown('<div class="main-header">Customer Churn Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Real-time risk scoring, predictive probability calibration, and automated retention playbooks powered by the verified Random Forest model backend.</div>',
    unsafe_allow_html=True,
)

# Customer Input Form
with st.form("churn_prediction_form"):
    st.subheader("📋 Customer Account & Service Profile")
    col1, col2, col3 = st.columns(3)

    # Column 1: Demographics & Account Identity
    with col1:
        st.markdown("**1. Demographics & Tenure**")
        gender = st.selectbox("Gender", ["Female", "Male"], index=0 if default_values["gender"] == "Female" else 1)
        senior = st.selectbox("Senior Citizen", [0, 1], index=default_values["SeniorCitizen"], format_func=lambda x: "Yes (65+)" if x == 1 else "No (<65)")
        partner = st.selectbox("Partner", ["Yes", "No"], index=0 if default_values["Partner"] == "Yes" else 1)
        dependents = st.selectbox("Dependents", ["Yes", "No"], index=0 if default_values["Dependents"] == "Yes" else 1)
        tenure = st.number_input("Tenure (Months Subscribed)", min_value=0, max_value=120, value=default_values["tenure"], step=1)

    # Column 2: Subscribed Services
    with col2:
        st.markdown("**2. Product & Service Add-ons**")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0 if default_values["PhoneService"] == "Yes" else 1)
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(default_values["MultipleLines"]))
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(default_values["InternetService"]))
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_values["OnlineSecurity"]))
        backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_values["OnlineBackup"]))
        device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_values["DeviceProtection"]))
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_values["TechSupport"]))
        stream_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_values["StreamingTV"]))
        stream_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_values["StreamingMovies"]))

    # Column 3: Contract & Financials
    with col3:
        st.markdown("**3. Contract & Financial Terms**")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(default_values["Contract"]))
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if default_values["PaperlessBilling"] == "Yes" else 1)
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            index=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ].index(default_values["PaymentMethod"]),
        )
        monthly_charges = st.number_input("Monthly Charges ($ USD)", min_value=0.0, max_value=500.0, value=float(default_values["MonthlyCharges"]), step=0.5)

        # Estimated total charges helper
        calc_total = float(round(tenure * monthly_charges, 2)) if tenure > 0 else 0.0
        total_charges = st.number_input(
            "Total Lifetime Charges ($ USD)",
            min_value=0.0,
            max_value=20000.0,
            value=float(default_values["TotalCharges"]) if default_values["TotalCharges"] > 0 else calc_total,
            step=10.0,
            help="Total cumulative amount charged. For new subscribers (tenure=0), this is 0.0.",
        )

    submit_button = st.form_submit_button("⚡ Predict Churn Risk", type="primary", use_container_width=True)

# Prediction Result Display
if submit_button:
    if not backend_healthy:
        st.error("❌ Cannot communicate with FastAPI backend. Ensure the server is running on the configured URL.")
    else:
        payload = {
            "gender": gender,
            "SeniorCitizen": int(senior),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device_prot,
            "TechSupport": tech_support,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
        }

        with st.spinner("Submitting customer profile to FastAPI inference engine..."):
            try:
                resp = requests.post(f"{api_base_url}/predict", json=payload, timeout=5.0)

                if resp.status_code == 200:
                    result = resp.json()
                    prob = result["churn_probability"]
                    pred_label = result["churn_label"]
                    risk_tier = result["risk_tier"]
                    threshold = result["classification_threshold"]
                    recommendation = result["retention_recommendation"]

                    st.markdown("---")
                    st.subheader("🔍 Prediction Results & Risk Assessment")

                    # Metric Summary Cards
                    m1, m2, m3 = st.columns(3)

                    # Card 1: Probability
                    with m1:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div style="color: #64748B; font-weight: 600;">CHURN PROBABILITY</div>
                                <div class="metric-value">{prob * 100:.1f}%</div>
                                <div style="color: #94A3B8; font-size: 0.85rem;">Calibrated Ensemble Posterior</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Card 2: Outcome Decision
                    with m2:
                        decision_color = "#DC2626" if pred_label == "Churn" else "#16A34A"
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div style="color: #64748B; font-weight: 600;">PREDICTED OUTCOME</div>
                                <div class="metric-value" style="color: {decision_color};">{pred_label.upper()}</div>
                                <div style="color: #94A3B8; font-size: 0.85rem;">Decision Threshold: <b>{threshold:.2f}</b></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Card 3: Risk Stratification Tier
                    with m3:
                        badge_class = "badge-high" if risk_tier == "High Risk" else ("badge-medium" if risk_tier == "Medium Risk" else "badge-low")
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div style="color: #64748B; font-weight: 600;">ACCOUNT RISK TIER</div>
                                <div style="margin: 0.7rem 0;"><span class="{badge_class}">{risk_tier.upper()}</span></div>
                                <div style="color: #94A3B8; font-size: 0.85rem;">Audit Tier Classification</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Visual Probability Progress Bar
                    st.markdown("##### 📈 Risk Scale Indicator")
                    st.progress(float(prob))

                    # SHAP Explainability Section (Genuine Local Attributions from Backend)
                    st.markdown("##### 🧠 Local Model Explainability (SHAP Values)")
                    explanation = result.get("explanation")
                    if explanation:
                        base_val = explanation.get("base_value", 0.50)
                        top_inc = explanation.get("top_churn_drivers", [])
                        top_dec = explanation.get("top_retention_factors", [])
                        summary_text = explanation.get("summary", "")

                        st.markdown(
                            f"""
                            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 1rem;">
                                <span style="color: #475569; font-size: 0.92rem;">
                                    <strong>How this prediction was formed:</strong> The ensemble model baseline probability prior to observing features is <strong>{base_val * 100:.1f}%</strong>. 
                                    Individual customer attributes either increase (push toward churn) or decrease (anchor retention) this probability, arriving at the final score of <strong>{prob * 100:.1f}%</strong>.
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Horizontal Feature Contribution Chart
                        combined_drivers = top_dec[::-1] + top_inc
                        if combined_drivers:
                            feats = [d["feature"].replace("_", " ") for d in combined_drivers]
                            vals = [d["shap_value"] * 100 for d in combined_drivers]
                            bar_colors = ["#10B981" if v < 0 else "#EF4444" for v in vals]

                            fig, ax = plt.subplots(figsize=(10, max(4.0, len(combined_drivers) * 0.42)), dpi=120)
                            bars = ax.barh(feats, vals, color=bar_colors, height=0.6, edgecolor="none")
                            ax.axvline(0, color="#64748B", linestyle="--", linewidth=0.8, alpha=0.7)

                            for bar, val in zip(bars, vals):
                                x_pos = bar.get_width()
                                ha = "left" if val >= 0 else "right"
                                offset = 0.2 if val >= 0 else -0.2
                                ax.annotate(
                                    f"{val:+.1f}%",
                                    (x_pos + offset, bar.get_y() + bar.get_height() / 2),
                                    ha=ha,
                                    va="center",
                                    fontsize=8.5,
                                    fontweight="bold",
                                    color="#DC2626" if val >= 0 else "#059669",
                                )

                            ax.set_xlabel("Impact on Churn Probability (Percentage Points)", fontsize=9.5, fontweight="bold", color="#1E293B")
                            ax.set_title("Local Feature Attributions for this Customer (SHAP)", fontsize=11.5, fontweight="bold", pad=10, color="#1E293B")
                            ax.grid(axis="x", linestyle="--", alpha=0.5)
                            ax.spines["top"].set_visible(False)
                            ax.spines["right"].set_visible(False)
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)

                        # Top 5 Positive and Negative Attribution Columns
                        shap_col1, shap_col2 = st.columns(2)
                        with shap_col1:
                            st.markdown("🔴 **Top Factors Increasing Churn Risk (+):**")
                            if top_inc:
                                for driver in top_inc:
                                    desc = driver.get("description") or f"{driver['feature']} adds +{driver['shap_value'] * 100:.1f}% to churn risk."
                                    st.markdown(f"- **{driver['feature']}** (`+{driver['shap_value'] * 100:.1f}%`): {desc}")
                            else:
                                st.caption("No significant positive risk drivers identified.")

                        with shap_col2:
                            st.markdown("🟢 **Top Factors Decreasing Churn Risk (-):**")
                            if top_dec:
                                for driver in top_dec:
                                    desc = driver.get("description") or f"{driver['feature']} reduces churn risk by {driver['shap_value'] * 100:.1f}%."
                                    st.markdown(f"- **{driver['feature']}** (`{driver['shap_value'] * 100:.1f}%`): {desc}")
                            else:
                                st.caption("No significant retention factors identified.")

                        # Summary Callout
                        if summary_text:
                            st.info(f"💡 **SHAP Executive Summary:** {summary_text}")
                    elif "shap_values" in result or "shap_explanation" in result:
                        shap_data = result.get("shap_values") or result.get("shap_explanation")
                        st.json(shap_data)
                    else:
                        st.caption(
                            "ℹ️ *Local SHAP attribution values are not returned in the backend response payload. "
                            "Global SHAP feature importance can be inspected in the model audit artifacts (`models/plots/shap_summary.png`).*"
                        )

                    # Retention Playbook Recommendation
                    st.markdown("##### 🛡️ Prescriptive Retention Playbook")
                    st.markdown(
                        f"""
                        <div class="playbook-box">
                            <strong style="color: #1E3A8A; font-size: 1.05rem;">Recommended Strategic Action:</strong><br>
                            <span style="color: #1E293B; line-height: 1.6;">{recommendation}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Key Drivers Checklist
                    st.markdown("##### ⚠️ Observed Risk Factor Analysis")
                    factors_col1, factors_col2 = st.columns(2)
                    with factors_col1:
                        if contract == "Month-to-month":
                            st.warning("⚠️ **Contract Vulnerability:** Customer is on a Month-to-month plan (42.7% baseline churn).")
                        else:
                            st.success(f"✅ **Contract Protection:** Long-term commitment ({contract}) significantly reduces churn likelihood.")

                        if payment == "Electronic check":
                            st.warning("⚠️ **Payment Friction:** Electronic check exhibits 45.3% churn rate.")
                        else:
                            st.success("✅ **Automated Payment:** Customer uses lower-friction automated payment.")

                    with factors_col2:
                        if tenure <= 6:
                            st.warning(f"⚠️ **Onboarding Cliff:** Low tenure ({tenure} months). Customers in months 1–6 face the highest attrition risk.")
                        else:
                            st.success(f"✅ **Tenure Stability:** Customer has {tenure} months of tenure.")

                        if internet == "Fiber optic" and (tech_support == "No" or security == "No"):
                            st.warning("⚠️ **Service Vulnerability:** High-speed Fiber Optic without TechSupport or OnlineSecurity add-ons.")

                    # Technical Payload Inspector Expander
                    with st.expander("🛠️ Inspect REST API Request & Response"):
                        st.json({"request_payload": payload, "response_data": result})

                elif resp.status_code == 422:
                    st.error("Validation Error: One or more input values are invalid.")
                    st.json(resp.json())
                else:
                    st.error(f"Backend error ({resp.status_code}): {resp.text}")

            except requests.exceptions.RequestException as exc:
                st.error(f"Failed to communicate with FastAPI backend: {exc}")
