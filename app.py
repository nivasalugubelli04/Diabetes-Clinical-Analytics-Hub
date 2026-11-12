import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page configuration with clean layout theme
st.set_page_config(page_title="Diabetes Clinical Analytics Hub", page_icon="🩺", layout="wide")

# Custom CSS injection for cards and styling
st.markdown("""<style>
    .main {
        background-color: #f7f9fc;
    }
    .reportview-container {
        background: #f7f9fc;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #eef2f6;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        margin-bottom: 15px;
    }
    .status-banner {
        padding: 24px;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load("diabetes_rf_model.joblib")
    scaler = joblib.load("diabetes_scaler.joblib")
    return model, scaler

try:
    model, scaler = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    st.error(f"Error loading serialized model artifacts: {e}")
    artifacts_loaded = False

# Top Header Banner
st.title("🩺 Diabetes Clinical Analytics Hub")
st.markdown("##### Advanced Predictive Diagnostics & Patient Biometrics Dashboard")
st.write("---")

# Sidebar for user inputs
with st.sidebar:
    st.header("🧪 Patient Clinical Parameters")
    st.write("Adjust patient biometrics below:")
    preg = st.slider("Pregnancies", min_value=0, max_value=17, value=2, step=1)
    glucose = st.slider("Glucose (mg/dL)", min_value=40.0, max_value=200.0, value=120.0, step=1.0)
    bp = st.slider("Blood Pressure (mmHg)", min_value=24.0, max_value=122.0, value=70.0, step=1.0)
    skin = st.slider("Skin Thickness (mm)", min_value=7.0, max_value=99.0, value=25.0, step=1.0)
    insulin = st.slider("Insulin (mu U/ml)", min_value=14.0, max_value=846.0, value=120.0, step=5.0)
    bmi = st.slider("BMI", min_value=18.0, max_value=67.0, value=30.0, step=0.1)
    pedigree = st.slider("Pedigree Function", min_value=0.08, max_value=2.42, value=0.47, step=0.01)
    age = st.slider("Age (years)", min_value=21, max_value=81, value=33, step=1)

# Main panel layout
col1, col2, col3 = st.columns([1, 1, 1.2])

if artifacts_loaded:
    # Prepare raw data payload
    raw_data = {
        'Pregnancies': preg,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': pedigree,
        'Age': age
    }

    df_input = pd.DataFrame([raw_data])
    df_input['Glucose_BMI'] = df_input['Glucose'] * df_input['BMI']
    df_input['Glucose_Age'] = df_input['Glucose'] * df_input['Age']

    feature_names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
        'Glucose_BMI', 'Glucose_Age'
    ]
    df_processed = df_input[feature_names]
    scaled_features = scaler.transform(df_processed)

    # Predict probability and class
    prob = float(model.predict_proba(scaled_features)[0][1])
    is_diabetic = int(model.predict(scaled_features)[0])

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("📊 Patient Profile")
        st.metric(label="Glucose Level", value=f"{glucose} mg/dL")
        st.metric(label="BMI Index", value=f"{bmi}")
        st.metric(label="Patient Age", value=f"{age} yrs")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("📈 Risk Metrics")
        st.metric(label="Risk Score", value=f"{prob * 100:.1f}%")
        st.metric(label="Pedigree Factor", value=f"{pedigree}")
        st.metric(label="Blood Pressure", value=f"{bp} mmHg")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.subheader("🎯 Diagnostic Conclusion")
        status_text = "DIABETIC / HIGH RISK" if is_diabetic == 1 else "NON-DIABETIC / LOW RISK"
        status_bg = "#e74c3c" if is_diabetic == 1 else "#2ecc71"

        st.markdown(f"""
        <div class="status-banner" style="background-color: {status_bg};">
            <h2 style="color: white; margin: 0;">{status_text}</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 16px;">
                Confidence Probability: {prob * 100:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Plot Gauge Chart
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.barh(['Risk Score'], [prob], color=status_bg, height=0.4, edgecolor='black')
        ax.set_xlim(0, 1.0)
        ax.axvline(0.5, color='#7f8c8d', linestyle='--', alpha=0.8, label='Decision Boundary (0.50)')
        ax.set_xlabel('Probability Index', fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.legend(loc='upper right', fontsize=8)
        plt.title("Inference Engine Risk Scale Output", fontsize=10, fontweight='bold', pad=10)
        plt.tight_layout()
        st.pyplot(fig)
else:
    st.warning("Please verify that the required .joblib model files are present in the app working directory.")
