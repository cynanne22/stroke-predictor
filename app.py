import streamlit as st
import pandas as pd
import joblib
import time

# =======================
# 1. Configuration & CSS (Medical Navy Theme)
# =======================
st.set_page_config(page_title="CerebroCare", layout="centered")

st.markdown("""
    <style>
        /* Import Font */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

        /* -- Background & Text Colors (Medical Navy Theme) -- */
        body {
            background-color: #0f172a; /* Dark Navy */
            color: #e2e8f0; /* Light Gray Text */
            font-family: 'Roboto', sans-serif;
        }
        
        .stApp {
            background-color: #0f172a;
        }

        /* -- Title Styling -- */
        .main-title {
            font-size: 3.5rem;
            font-weight: 700;
            color: #38bdf8; /* Light Blue / Cyan */
            text-align: center;
            margin-bottom: 10px;
            font-family: 'Roboto', sans-serif;
        }
        
        .sub-title {
            font-size: 1.2rem;
            color: #94a3b8;
            text-align: center;
            margin-bottom: 40px;
        }

        /* -- Headers -- */
        h1, h2, h3, .stHeader, .stSubheader {
            color: #f1f5f9 !important;
            font-family: 'Roboto', sans-serif;
        }

        /* -- Custom Buttons -- */
        .stButton>button {
            background-color: #0ea5e9; /* Sky Blue */
            color: white;
            font-size: 18px;
            padding: 10px 24px;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            width: 100%;
            transition: 0.3s;
        }

        .stButton>button:hover {
            background-color: #0284c7; /* Darker Blue on Hover */
        }

        /* -- Input Fields Styling -- */
        .stSelectbox, .stNumberInput, .stTextInput>div>input {
            background-color: #1e293b; /* Slate Blue */
            color: white;
            border-radius: 8px;
            border: 1px solid #334155;
        }
        
        /* -- Selectbox Dropdown Text -- */
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #1e293b;
            color: white;
        }
        
        /* -- Metrics & Success/Error -- */
        div[data-testid="stMetricValue"] {
            color: #38bdf8;
        }
        
        .stAlert {
            background-color: #1e293b;
            color: white;
            border: 1px solid #334155;
        }
    </style>
""", unsafe_allow_html=True)

# =======================
# 2. Load Model & Define Columns
# =======================
try:
    model = joblib.load("best_model.joblib")
except FileNotFoundError:
    st.error("Error: 'best_model.joblib' not found. Please upload it.")
    st.stop()

# ⚠ CRITICAL: This list must match your training data columns EXACTLY.
# If your model was trained with different columns, update this list.
MODEL_COLUMNS = [
    "age", "hypertension", "heart_disease", "ever_married", "avg_glucose_level", 
    "bmi", "gender_Male", "work_type_Never_worked", "work_type_Private", 
    "work_type_Self-employed", "work_type_children", "Residence_type_Urban", 
    "smoking_status_formerly smoked", "smoking_status_never smoked", "smoking_status_smokes"
]

# =======================
# 3. Session State (Navigation Memory)
# =======================
if 'prediction_done' not in st.session_state:
    st.session_state['prediction_done'] = False
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Home"
if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None
if 'prediction_proba' not in st.session_state:
    st.session_state['prediction_proba'] = 0.0

# =======================
# 4. Page Functions
# =======================

def show_home():
    st.markdown('<div class="main-title">CerebroCare</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-Powered Stroke Risk Assessment</div>', unsafe_allow_html=True)
    
    st.write("### Welcome")
    st.write("""
    **CerebroCare** is an advanced tool designed to predict stroke probability based on patient medical history and lifestyle factors.
    
    Use the sidebar menu or click below to start an assessment.
    """)
    
    if st.button("Start Assessment"):
        st.session_state['current_page'] = "Prediction"
        st.rerun()

def show_prediction():
    st.markdown('<div class="main-title">CerebroCare</div>', unsafe_allow_html=True)
    st.write("### Patient Data Entry")
    
    col1, col2 = st.columns(2)

    # ===== Left column =====
    with col1:
        st.subheader("Patient Info")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        ever_married = st.selectbox("Ever Married?", ["Yes", "No"])
        residence = st.selectbox("Residence Type", ["Urban", "Rural"])
        bmi = st.number_input("BMI", min_value=0.0, value=25.0)

    # ===== Right column =====
    with col2:
        st.subheader("Medical History")
        hypertension = st.selectbox("Hypertension", ["Yes", "No"])
        heart_disease = st.selectbox("Heart Disease", ["Yes", "No"])
        smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Never_worked", "children", "Govt_job"])
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=90.0)

    st.markdown("---")

    # ===== Prediction Logic =====
    if st.button("Analyze Risk Profile"):
        with st.spinner("Processing data..."):
            time.sleep(1) # Simulated loading effect
            
            # 1. Create Input Dictionary
            input_dict = {
                "age": age,
                "hypertension": 1 if hypertension == "Yes" else 0,
                "heart_disease": 1 if heart_disease == "Yes" else 0,
                "ever_married": 1 if ever_married == "Yes" else 0,
                "avg_glucose_level": avg_glucose_level,
                "bmi": bmi,
                "gender_Male": 1 if gender == "Male" else 0,
                "work_type_Never_worked": 1 if work_type == "Never_worked" else 0,
                "work_type_Private": 1 if work_type == "Private" else 0,
                "work_type_Self-employed": 1 if work_type == "Self-employed" else 0,
                "work_type_children": 1 if work_type == "children" else 0,
                "Residence_type_Urban": 1 if residence == "Urban" else 0,
                "smoking_status_formerly smoked": 1 if smoking_status == "formerly smoked" else 0,
                "smoking_status_never smoked": 1 if smoking_status == "never smoked" else 0,
                "smoking_status_smokes": 1 if smoking_status == "smokes" else 0,
            }

            # 2. Convert to Safe DataFrame (Prevents Column Mismatch Crashes)
            input_df = pd.DataFrame([input_dict])
            final_df = input_df.reindex(columns=MODEL_COLUMNS, fill_value=0)

            # 3. Predict
            prediction = model.predict(final_df)[0]
            probability = model.predict_proba(final_df)[0][1]

            # 4. Save to Session State & Navigate
            st.session_state['prediction_result'] = prediction
            st.session_state['prediction_proba'] = probability
            st.session_state['prediction_done'] = True
            st.session_state['current_page'] = "Analysis Result"
            st.rerun()

def show_result():
    st.markdown('<div class="main-title">CerebroCare</div>', unsafe_allow_html=True)
    st.write("### Analysis Result")
    
    prob = st.session_state['prediction_proba']
    res = st.session_state['prediction_result']

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(label="Stroke Probability", value=f"{prob:.1%}")
    
    with col2:
        if res == 1:
            st.error("⚠ **High Stroke Risk Detected**")
            st.write("The model indicates a significant correlation with stroke risk factors. Please consult a medical professional.")
        else:
            st.success("🟢 **Low Stroke Risk Detected**")
            st.write("Your health metrics indicate a low probability of stroke. Maintain a healthy lifestyle.")

    st.markdown("---")
    if st.button("Start New Assessment"):
        st.session_state['prediction_done'] = False
        st.session_state['current_page'] = "Prediction"
        st.rerun()

# =======================
# 5. Main Navigation Controller
# =======================
def main():
    # Sidebar Navigation
    st.sidebar.header("Navigation")
    
    menu_options = ["Home", "Prediction"]
    if st.session_state['prediction_done']:
        menu_options.append("Analysis Result")
    
    # Using Key to auto-update session state
    selection = st.sidebar.selectbox("Go to", menu_options, key='current_page')

    if selection == "Home":
        show_home()
    elif selection == "Prediction":
        show_prediction()
    elif selection == "Analysis Result":
        show_result()

if __name__ == "__main__":
    main()
