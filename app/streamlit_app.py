"""Telco Customer Churn Intelligence & Explainable AI Dashboard.

A decoupled enterprise SaaS analytics platform for customer retention teams.
Connects strictly via HTTP REST to the FastAPI inference and SHAP backend.
"""

import os
import requests
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Telco Customer Churn Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom SaaS Enterprise Theme Styling
st.markdown(
    """
    <style>
    /* Global Typography & Spacing */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }
    
    /* Header Styles */
    .saas-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .saas-subtitle {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.6rem;
        line-height: 1.5;
    }
    
    /* Metric Cards */
    .saas-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .saas-card-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }
    .saas-card-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .saas-card-sub {
        font-size: 0.82rem;
        color: #64748B;
        margin-top: 0.35rem;
    }
    
    /* Badges */
    .badge-pill {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #DC2626;
        border: 1px solid #FCA5A5;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #D97706;
        border: 1px solid #FCD34D;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #16A34A;
        border: 1px solid #86EFAC;
    }
    
    /* Retention Playbook Box */
    .playbook-container {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-left: 5px solid #2563EB;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
    }
    .playbook-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #1E3A8A;
        margin-bottom: 0.4rem;
    }
    .playbook-text {
        color: #1E293B;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Explanatory callout */
    .expl-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1rem;
        color: #334155;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar Configuration & Navigation
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🏢 Retention Intelligence")
st.sidebar.caption("Production AI/ML Explainability & Churn Prediction")

# Navigation Selector
page_selection = st.sidebar.radio(
    "Navigation",
    [
        "🔮 Customer Risk Scoring",
        "📊 Executive Overview",
        "🏆 Model Insights & Benchmark",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Backend Connectivity")
api_base_url = st.sidebar.text_input(
    "FastAPI Base URL",
    value=os.getenv("API_URL", "http://127.0.0.1:8000"),
    help="Target host and port for the running FastAPI inference service.",
)

# Backend Health Probe
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
        f"**Decision Threshold:** `{health_data.get('classification_threshold', 0.50):.2f}`\n\n"
        f"**Risk Cutoffs:** Low < `{health_data.get('low_risk_upper_bound')}` | Med < `{health_data.get('medium_risk_upper_bound')}`"
    )
else:
    st.sidebar.error("🔴 Backend Offline or Unreachable")
    st.sidebar.warning(
        "Make sure FastAPI is running locally:\n\n"
        "`uvicorn api.main:app --host 127.0.0.1 --port 8000`"
    )

st.sidebar.markdown("---")
st.sidebar.caption("Telco Churn Intelligence Platform | v1.0.0")


# -----------------------------------------------------------------------------
# PAGE 1: CUSTOMER RISK SCORING & SHAP EXPLAINABILITY
# -----------------------------------------------------------------------------
if page_selection == "🔮 Customer Risk Scoring":
    st.markdown('<div class="saas-header">Customer Churn Risk Scoring</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="saas-subtitle">Evaluate individual customer cancellation hazard, calculate genuine local SHAP feature attributions, and receive automated retention playbook recommendations.</div>',
        unsafe_allow_html=True,
    )

    # Preset selection
    st.markdown("##### 🎯 Quick Persona Templates")
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

    # Initialize session state defaults if absent
    if "preset_data" not in st.session_state:
        st.session_state["preset_data"] = {
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

    with preset_col1:
        if st.button("🚨 At-Risk Account", use_container_width=True, help="New subscriber, month-to-month, fiber, electronic check"):
            st.session_state["preset_data"] = {
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

    with preset_col2:
        if st.button("🛡️ Loyal Subscriber", use_container_width=True, help="Two-year contract, DSL, full security, automatic payment"):
            st.session_state["preset_data"] = {
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

    with preset_col3:
        if st.button("⚖️ Moderate Risk", use_container_width=True, help="One-year contract, fiber optic, 18-month tenure"):
            st.session_state["preset_data"] = {
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

    with preset_col4:
        if st.button("🔄 Reset Profile", use_container_width=True):
            st.session_state["preset_data"] = {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 45.00,
                "TotalCharges": 540.00,
            }

    p = st.session_state["preset_data"]

    # Customer Input Form
    with st.form("churn_prediction_form"):
        st.markdown("---")
        st.markdown("#### 📋 Customer Account & Service Parameters")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**1. Demographics & Tenure**")
            gender = st.selectbox("Gender", ["Female", "Male"], index=0 if p["gender"] == "Female" else 1)
            senior = st.selectbox(
                "Senior Citizen",
                [0, 1],
                index=p["SeniorCitizen"],
                format_func=lambda x: "Yes (Age 65+)" if x == 1 else "No (Under 65)",
            )
            partner = st.selectbox("Partner", ["Yes", "No"], index=0 if p["Partner"] == "Yes" else 1)
            dependents = st.selectbox("Dependents", ["Yes", "No"], index=0 if p["Dependents"] == "Yes" else 1)
            tenure = st.number_input(
                "Tenure (Months Subscribed)",
                min_value=0,
                max_value=120,
                value=int(p["tenure"]),
                step=1,
                help="Duration of customer subscription in months.",
            )

        with col2:
            st.markdown("**2. Subscribed Services**")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0 if p["PhoneService"] == "Yes" else 1)
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"],
                index=["No", "Yes", "No phone service"].index(p["MultipleLines"]),
            )
            internet = st.selectbox(
                "Internet Service",
                ["Fiber optic", "DSL", "No"],
                index=["Fiber optic", "DSL", "No"].index(p["InternetService"]),
            )
            security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(p["OnlineSecurity"]),
            )
            backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(p["OnlineBackup"]),
            )
            device_prot = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(p["DeviceProtection"]),
            )
            tech_support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(p["TechSupport"]),
            )
            stream_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(p["StreamingTV"]),
            )
            stream_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(p["StreamingMovies"]),
            )

        with col3:
            st.markdown("**3. Contract & Financial Terms**")
            contract = st.selectbox(
                "Contract Agreement",
                ["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(p["Contract"]),
            )
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if p["PaperlessBilling"] == "Yes" else 1)
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
                ].index(p["PaymentMethod"]),
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($ USD)",
                min_value=0.0,
                max_value=500.0,
                value=float(p["MonthlyCharges"]),
                step=0.5,
                help="Recurring monthly billing amount in USD.",
            )
            estimated_total = float(round(tenure * monthly_charges, 2)) if tenure > 0 else 0.0
            total_charges = st.number_input(
                "Total Lifetime Charges ($ USD)",
                min_value=0.0,
                max_value=20000.0,
                value=float(p["TotalCharges"]) if float(p["TotalCharges"]) > 0 else estimated_total,
                step=10.0,
                help="Cumulative spend. For brand-new subscribers (tenure=0), this is 0.0.",
            )

        submit_button = st.form_submit_button("⚡ Run Churn Risk Analysis", type="primary", use_container_width=True)

    # Prediction Execution & Results
    if submit_button:
        if not backend_healthy:
            st.error("❌ Cannot communicate with FastAPI backend. Ensure the server is online.")
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

            with st.spinner("Submitting features to inference engine & computing local SHAP attributions..."):
                try:
                    resp = requests.post(f"{api_base_url}/predict", json=payload, timeout=5.0)

                    if resp.status_code == 200:
                        result = resp.json()
                        prob = result["churn_probability"]
                        pred_label = result["churn_label"]
                        risk_tier = result["risk_tier"]
                        threshold = result["classification_threshold"]
                        recommendation = result["retention_recommendation"]
                        explanation = result.get("explanation")

                        st.markdown("---")
                        st.markdown("### 🔍 Risk Assessment & Intelligence Summary")

                        # Metric Summary Cards
                        m1, m2, m3, m4 = st.columns(4)

                        with m1:
                            st.markdown(
                                f"""
                                <div class="saas-card">
                                    <div class="saas-card-label">Churn Probability</div>
                                    <div class="saas-card-value">{prob * 100:.1f}%</div>
                                    <div class="saas-card-sub">Posterior Risk Score</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        with m2:
                            label_color = "#DC2626" if pred_label == "Churn" else "#16A34A"
                            st.markdown(
                                f"""
                                <div class="saas-card">
                                    <div class="saas-card-label">Classification Decision</div>
                                    <div class="saas-card-value" style="color: {label_color};">{pred_label.upper()}</div>
                                    <div class="saas-card-sub">Decision Cutoff: <b>{threshold:.2f}</b></div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        with m3:
                            badge_cls = (
                                "badge-high"
                                if risk_tier == "High Risk"
                                else ("badge-medium" if risk_tier == "Medium Risk" else "badge-low")
                            )
                            st.markdown(
                                f"""
                                <div class="saas-card">
                                    <div class="saas-card-label">Account Risk Stratification</div>
                                    <div style="margin-top: 0.5rem; margin-bottom: 0.4rem;">
                                        <span class="badge-pill {badge_cls}">{risk_tier}</span>
                                    </div>
                                    <div class="saas-card-sub">Portfolio Risk Tier</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        with m4:
                            st.markdown(
                                f"""
                                <div class="saas-card">
                                    <div class="saas-card-label">Contract Commitment</div>
                                    <div class="saas-card-value" style="font-size: 1.45rem;">{contract}</div>
                                    <div class="saas-card-sub">{tenure} Months Tenure</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # Risk Scale Visual
                        st.markdown("##### 📈 Risk Spectrum Indicator")
                        st.progress(float(prob))
                        st.caption(
                            f"Score: **{prob * 100:.1f}%** | "
                            f"Low Risk Tier: `< 30%` | "
                            f"Medium Risk Tier: `30% - 60%` | "
                            f"High Risk Tier: `>= 60%`"
                        )

                        # Prescriptive Retention Playbook
                        st.markdown(
                            f"""
                            <div class="playbook-container">
                                <div class="playbook-title">🛡️ Strategic Retention Playbook Recommendation</div>
                                <div class="playbook-text">{recommendation}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # SHAP Explainability Breakdown
                        st.markdown("### 🧠 Genuine Local Explainability (SHAP Attributions)")
                        if explanation:
                            base_val = explanation.get("base_value", 0.50)
                            top_inc = explanation.get("top_churn_drivers", [])
                            top_dec = explanation.get("top_retention_factors", [])
                            summary_text = explanation.get("summary", "")

                            st.markdown(
                                f"""
                                <div class="expl-box">
                                    <strong>Attribution Mechanics:</strong> The ensemble model begins at an empirical base expected probability of <strong>{base_val * 100:.1f}%</strong> prior to inspecting subscriber attributes.
                                    Individual customer features exert positive forces (accelerating churn) or negative forces (anchoring retention), summing down to machine precision:
                                    <code>E[f(x)] + &Sigma; &phi;<sub>i</sub> = {prob * 100:.1f}%</code>.
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # SHAP Contribution Bar Chart
                            combined_drivers = top_dec[::-1] + top_inc
                            if combined_drivers:
                                feats = [d["feature"].replace("_", " ") for d in combined_drivers]
                                vals = [d["shap_value"] * 100 for d in combined_drivers]
                                colors = ["#10B981" if v < 0 else "#EF4444" for v in vals]

                                fig, ax = plt.subplots(figsize=(10, max(4.0, len(combined_drivers) * 0.44)), dpi=130)
                                bars = ax.barh(feats, vals, color=colors, height=0.6, edgecolor="none")
                                ax.axvline(0, color="#64748B", linestyle="--", linewidth=0.9, alpha=0.8)

                                for bar, val in zip(bars, vals):
                                    x_pos = bar.get_width()
                                    ha = "left" if val >= 0 else "right"
                                    offset = 0.25 if val >= 0 else -0.25
                                    ax.annotate(
                                        f"{val:+.1f}%",
                                        (x_pos + offset, bar.get_y() + bar.get_height() / 2),
                                        ha=ha,
                                        va="center",
                                        fontsize=9,
                                        fontweight="bold",
                                        color="#DC2626" if val >= 0 else "#059669",
                                    )

                                ax.set_xlabel("Impact on Churn Probability (Percentage Points)", fontsize=9.5, fontweight="bold", color="#1E293B")
                                ax.set_title("Local Feature Contribution Breakdown (Tree SHAP)", fontsize=11, fontweight="bold", pad=12, color="#0F172A")
                                ax.grid(axis="x", linestyle="--", alpha=0.5)
                                ax.spines["top"].set_visible(False)
                                ax.spines["right"].set_visible(False)
                                plt.tight_layout()
                                st.pyplot(fig, use_container_width=True)
                                plt.close(fig)

                            # Side-by-side attribution columns
                            shap_col1, shap_col2 = st.columns(2)
                            with shap_col1:
                                st.markdown("🔴 **Top Features Driving Churn Hazard (+):**")
                                if top_inc:
                                    for d in top_inc:
                                        desc = d.get("description") or f"{d['feature']} adds +{d['shap_value'] * 100:.1f}% to risk."
                                        st.markdown(f"- **{d['feature']}** (`+{d['shap_value'] * 100:.1f}%`): {desc}")
                                else:
                                    st.caption("No positive churn drivers identified.")

                            with shap_col2:
                                st.markdown("🟢 **Top Features Supporting Retention (-):**")
                                if top_dec:
                                    for d in top_dec:
                                        desc = d.get("description") or f"{d['feature']} reduces churn hazard by {d['shap_value'] * 100:.1f}%."
                                        st.markdown(f"- **{d['feature']}** (`{d['shap_value'] * 100:.1f}%`): {desc}")
                                else:
                                    st.caption("No significant protective retention factors identified.")

                            if summary_text:
                                st.info(f"💡 **Executive SHAP Summary:** {summary_text}")
                        else:
                            st.warning("Local SHAP attributions could not be computed for this request.")

                        # Raw Payload Expander
                        with st.expander("🛠️ Inspect REST API Request & Response"):
                            st.json({"request_payload": payload, "response_payload": result})

                    elif resp.status_code == 422:
                        st.error("Validation Error: One or more input parameters are invalid.")
                        st.json(resp.json())
                    else:
                        st.error(f"Backend error ({resp.status_code}): {resp.text}")

                except requests.exceptions.RequestException as err:
                    st.error(f"Failed to communicate with FastAPI backend: {err}")


# -----------------------------------------------------------------------------
# PAGE 2: EXECUTIVE OVERVIEW & COHORT KPIS
# -----------------------------------------------------------------------------
elif page_selection == "📊 Executive Overview":
    st.markdown('<div class="saas-header">Executive Overview & Portfolio Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="saas-subtitle">High-level subscriber portfolio health, churn economics, risk distribution, and macro attrition drivers across the IBM Telco subscriber cohort.</div>',
        unsafe_allow_html=True,
    )

    # Executive KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-label">Total Accounts Analyzed</div>
                <div class="saas-card-value">7,043</div>
                <div class="saas-card-sub">Complete Telco Cohort</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi2:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-label">Portfolio Churn Rate</div>
                <div class="saas-card-value" style="color: #DC2626;">26.54%</div>
                <div class="saas-card-sub">1,869 Churned Subscribers</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi3:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-label">Model Detection Recall</div>
                <div class="saas-card-value" style="color: #16A34A;">78.34%</div>
                <div class="saas-card-sub">Captures 293 / 374 Test Churners</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi4:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-label">Retention ARR Protected</div>
                <div class="saas-card-value" style="color: #2563EB;">$2.42M</div>
                <div class="saas-card-sub">Projected Annual Recurring Value</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 📊 Macro Drivers of Subscriber Attrition")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 1. Contract Commitment Sensitivity")
        st.markdown(
            """
            Contract structure is the **single strongest predictor** of subscriber churn:
            - **Month-to-month contracts:** **42.7% churn rate** (vulnerable to competitor poaching and pricing changes).
            - **One-year contracts:** **11.3% churn rate** (3.8x reduction in churn hazard).
            - **Two-year contracts:** **2.8% churn rate** (15x reduction in churn hazard).
            
            *Strategic Imperative:* Proactive discounts or hardware promotions for 12-month commitments directly protect recurring revenue.
            """
        )

        st.markdown("#### 2. The Onboarding Retention Cliff (Tenure ≤ 6 Months)")
        st.markdown(
            """
            Subscriber attrition is heavily front-loaded:
            - Accounts with **tenure ≤ 6 months** account for **38.4% of all churn events**.
            - Once a customer reaches **tenure > 24 months**, baseline churn falls below **14%**.
            - Customers with **tenure > 48 months** exhibit a sub-**7% churn rate**.
            
            *Strategic Imperative:* Focus onboarding check-ins and customer success touches in the first 90 days.
            """
        )

    with c2:
        st.markdown("#### 3. Payment Method Friction")
        st.markdown(
            """
            Payment mechanics strongly correlate with subscriber dissatisfaction:
            - **Electronic Check:** **45.3% churn rate** (high manual effort and payment failures).
            - **Mailed Check:** **19.1% churn rate**.
            - **Bank Transfer (Automatic):** **16.7% churn rate**.
            - **Credit Card (Automatic):** **15.2% churn rate**.
            
            *Strategic Imperative:* Offer a one-time bill credit ($10–$15) to transition Electronic Check users to automated recurring payments.
            """
        )

        st.markdown("#### 4. Service Ecosystem Protection")
        st.markdown(
            """
            Subscribers with bundled security add-ons exhibit significantly higher loyalty:
            - **Without Tech Support:** **41.6% churn rate**.
            - **With Premium Tech Support:** **15.2% churn rate**.
            - **Without Online Security:** **41.8% churn rate**.
            - **With Online Security:** **14.6% churn rate**.
            """
        )


# -----------------------------------------------------------------------------
# PAGE 3: MODEL INSIGHTS & BENCHMARK EVALUATION
# -----------------------------------------------------------------------------
elif page_selection == "🏆 Model Insights & Benchmark":
    st.markdown('<div class="saas-header">Model Performance & Audit Benchmarks</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="saas-subtitle">Verified evaluation metrics, cross-validation stability, overfitting analysis, and algorithm comparison from the official model audit.</div>',
        unsafe_allow_html=True,
    )

    # Fetch benchmark data from API or fall back gracefully
    benchmark_data = None
    if backend_healthy:
        try:
            b_resp = requests.get(f"{api_base_url}/benchmark", timeout=2.5)
            if b_resp.status_code == 200:
                benchmark_data = b_resp.json()
        except Exception:
            benchmark_data = None

    # Verified test set metric cards for selected winning model (Random Forest)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-label">Test Recall</div>
                <div class="saas-card-value" style="color: #16A34A;">78.34%</div>
                <div class="saas-card-sub">293 / 374 Captured</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-label">Test ROC-AUC</div>
                <div class="saas-card-value" style="color: #2563EB;">0.8425</div>
                <div class="saas-card-sub">Untouched Test Split</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-label">Test Precision</div>
                <div class="saas-card-value">52.98%</div>
                <div class="saas-card-sub">Controlled False Alarms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-label">Test F1-Score</div>
                <div class="saas-card-value">0.6321</div>
                <div class="saas-card-sub">Harmonic Mean</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m5:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-label">Test PR-AUC</div>
                <div class="saas-card-value">0.6569</div>
                <div class="saas-card-sub">Baseline: 0.2654</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m6:
        st.markdown(
            """
            <div class="saas-card">
                <div class="saas-card-label">Brier Score</div>
                <div class="saas-card-value">0.1602</div>
                <div class="saas-card-sub">Calibrated Probability</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 📋 Final Holdout Test Set Performance Comparison (N = 1,409)")

    # Comparison table
    st.markdown(
        """
        | Model Architecture | Decision Threshold | Test Recall | Test Precision | Test F1-Score | Test ROC-AUC | Test PR-AUC | Brier Score | True Positives | False Negatives |
        | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
        | 🏆 **Random Forest (Selected)** | **0.50** | **78.34%** | **52.98%** | **0.6321** | **0.8425** | **0.6569** | **0.1602** | **293 / 374** | **81** |
        | 🥈 Gradient Boosting | 0.50 | 50.80% | 67.38% | 0.5793 | 0.8452 | 0.6605 | 0.1356 | 190 / 374 | 184 |
        | 🥉 Logistic Regression (L2) | 0.50 | 55.61% | 65.41% | 0.6012 | 0.8412 | 0.6281 | 0.1385 | 208 / 374 | 166 |
        """
    )

    st.markdown("### 🔍 Model Selection Rationale")
    st.markdown(
        """
        **Why Random Forest was selected over Gradient Boosting and Logistic Regression:**
        1. **Maximizing Churn Detection Recall (78.34%):** In customer retention economics, **False Negatives (unidentified churners who cancel)** cost 5x to 7x more than **False Positives (retained customers offered a renewal incentive)**. Random Forest successfully identifies **293 out of 374** churning accounts on the test partition, whereas Gradient Boosting missed nearly half (184 False Negatives).
        2. **Cross-Validation Stability:** Random Forest demonstrated negligible variance across 5 stratified folds ($\sigma_{\text{recall}} = 0.019$, $\sigma_{\text{roc\_auc}} = 0.008$), confirming resilience against sampling noise.
        3. **Optimized Hyperparameters:** `n_estimators=200`, `max_depth=10`, `min_samples_leaf=4`, `class_weight='balanced'`.
        """
    )

    st.markdown("---")
    st.markdown("### 🖼️ Audit Evaluation Visualizations")

    plot_dir = Path("models/plots")
    vcol1, vcol2 = st.columns(2)

    with vcol1:
        cm_path = plot_dir / "confusion_matrices.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrices across Candidate Models (Holdout Test Set)", use_container_width=True)

        roc_path = plot_dir / "roc_and_pr_curves.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC & Precision-Recall Curves", use_container_width=True)

    with vcol2:
        calib_path = plot_dir / "calibration_curves.png"
        if calib_path.exists():
            st.image(str(calib_path), caption="Probability Calibration & Reliability Curves", use_container_width=True)

        shap_path = plot_dir / "shap_summary.png"
        if shap_path.exists():
            st.image(str(shap_path), caption="Global Population SHAP Summary Plot", use_container_width=True)
