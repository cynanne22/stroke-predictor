import streamlit as st
import pandas as pd
import joblib
import time

# =======================
# 1. Configuration & CSS (CerebroCare Theme)
# =======================
st.set_page_config(page_title="CerebroCare", layout="centered")

st.markdown("""
    <style>
        /* Import Font */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

        /* -- BACKGROUND & TEXT -- */
        body {
            background-color: #0f172a; /* Dark Navy */
            color: #e2e8f0;
            font-family: 'Roboto', sans-serif;
        }
        
        .stApp {
            background-color: #0f172a;
        }

        /* -- HEADERS -- */
        .main-title {
            font-size: 3.5rem;
            font-weight: 700;
            color: #38bdf8; /* Cyan */
            text-align: center;
            margin-bottom: 20px;
        }
        
        h3, .stHeader, .stSubheader {
            color: #f1f5f9 !important;
            margin-top: 20px;
        }

        /* -- WIDGET LABELS (Age, Gender, etc.) -- */
        /* Pastikan label transparan, tanpa kotak, hanya teks putih */
        .stNumberInput label p, 
        .stSelectbox label p, 
        .stTextInput label p,
        div[data-testid="stWidgetLabel"] p {
            color: #f1f5f9 !important;
            font-size: 15px !important;
            font-weight: 500 !important;
            background-color: transparent !important; /* Tidak ada background box */
            margin-bottom: 2px !important;
        }

        /* -- INPUT BOXES ONLY (Kotak Isian) -- */
        /* CSS ini hanya menargetkan kotak input, bukan labelnya */
        div[data-baseweb="select"] > div, 
        div[data-baseweb="input"] > div {
            background-color: #1e293b !important; /* Warna dalam kotak */
            color: white !important;
            border: 1px solid #334155 !important; /* Garis pinggir kotak */
            border-radius: 8px !important;
        }
        
        /* -- TEXT INSIDE INPUT -- */
        div[data-baseweb="select"] span, input {
            color: white !important;
        }

        /* -- DROPDOWN MENU -- */
        div[data-baseweb="popover"] div {
            background-color: #1e293b !important;
            color: white !important;
        }

        /* -- BUTTONS -- */
        .stButton>button {
            background-color: #0ea5e9;
            color: white;
            font-size: 18px;
            padding: 12px 0;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            width: 100%;
            margin-top: 20px;
        }
        .stButton>button:hover {
            background-color: #0284c7;
        }
        
        /* -- ALERTS -- */
        .stAlert {
            background-color: #1e293b;
            color: white;
            border: 1px solid #334155;
        }
    </style>
""", unsafe_allow_html=True)


# =======================
# 2. Load Model
# =======================
try:
    model = joblib.load("best_model.joblib")
except FileNotFoundError:
    st.error("Error: 'best_model.joblib' not found.")
    st.stop()

# Columns harus sesuai training data
MODEL_COLUMNS = [
    "age", "hypertension", "heart_disease", "ever_married", "avg_glucose_level", 
    "bmi", "gender_Male", "work_type_Never_worked", "work_type_Private", 
    "work_type_Self-employed", "work_type_children", "Residence_type_Urban", 
    "smoking_status_formerly smoked", "smoking_status_never smoked", "smoking_status_smokes"
]

# =======================
# 3. MAIN APP
# =======================
def main():
    # Logo & Title
    # col1, col2, col3 = st.columns([1, 2, 1]) 
    # with col2:
    #      st.image("cerebrocarelogo.png", use_column_width=True) 

    st.markdown('<div class="main-title">CerebroCare</div>', unsafe_allow_html=True)
    
    st.write("""
        ### AI-Powered Stroke Risk Assessment
        Please fill out the patient details below to analyze the risk profile.
    """)
    st.divider()

    # ==========================================
    # FORM INPUT (Disusun Vertikal / 1 Kolom)
    # ==========================================
    
    # --- BAGIAN 1: PATIENT INFO ---
    st.subheader("Patient Info")
    
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    ever_married = st.selectbox("Ever Married?", ["Yes", "No"])
    residence = st.selectbox("Residence Type", ["Urban", "Rural"])
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)

    st.markdown("<br>", unsafe_allow_html=True) # Jarak visual

    # --- BAGIAN 2: MEDICAL HISTORY (Di Bawahnya) ---
    st.subheader("Medical History")
    
    hypertension = st.selectbox("Hypertension", ["Yes", "No"])
    heart_disease = st.selectbox("Heart Disease", ["Yes", "No"])
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Never_worked", "children", "Govt_job"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=90.0)

    st.markdown("---")

    # =======================
    # PREDICT BUTTON
    # =======================
    if st.button("Analyze Stroke Risk"):
        
        with st.spinner("Analyzing patient data..."):
            time.sleep(0.5) 
            
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

            # Safety check dataframe
            input_df = pd.DataFrame([input_dict])
            final_df = input_df.reindex(columns=MODEL_COLUMNS, fill_value=0)

            # Prediction
            prediction = model.predict(final_df)[0]
            probability = model.predict_proba(final_df)[0][1]

            st.subheader("Assessment Result")

            if prediction == 1:
                st.error(f"⚠ High Stroke Risk Detected\n\nProbability: {probability:.2%}")
                st.write("Recommendation: Consult a medical professional immediately.")
            else:
                st.success(f"🟢 Low Stroke Risk Detected\n\nProbability: {probability:.2%}")
                st.write("Recommendation: Maintain a healthy lifestyle.")

if __name__ == "__main__":
    main()
