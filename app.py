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

        /* -- MAIN BODY COLORS -- */
        body {
            background-color: #0f172a; /* Dark Navy */
            color: #e2e8f0; /* Light Gray Text */
            font-family: 'Roboto', sans-serif;
        }
        
        .stApp {
            background-color: #0f172a;
        }

        /* -- TITLE STYLING -- */
        .main-title {
            font-size: 3.5rem;
            font-weight: 700;
            color: #38bdf8; /* Cyan/Light Blue */
            text-align: center;
            margin-bottom: 10px;
            font-family: 'Roboto', sans-serif;
        }
        
        h3, .stHeader, .stSubheader {
            color: #f1f5f9 !important; /* White-ish for headers */
        }

        /* -- INPUT FIELDS -- */
        /* Making inputs lighter so text is readable */
        .stSelectbox, .stNumberInput, .stTextInput>div>input {
            background-color: #1e293b; /* Slate Blue */
            color: white;
            border-radius: 8px;
            border: 1px solid #334155;
        }
        
        /* Dropdown text fix */
        div[data-baseweb="select"] > div {
            background-color: #1e293b !important;
            color: white !important;
            border-color: #334155 !important;
        }
        div[data-baseweb="select"] span {
            color: white !important;
        }
        div[data-baseweb="popover"] div {
            background-color: #1e293b !important;
            color: white !important;
        }
        
        /* -- BUTTONS -- */
        .stButton>button {
            background-color: #0ea5e9; /* Sky Blue */
            color: white;
            font-size: 18px;
            padding: 12px 20px;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            transition: 0.3s;
            width: 100%;
        }

        .stButton>button:hover {
            background-color: #0284c7; /* Darker Blue */
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
# 2. Load Trained Model
# =======================
try:
    model = joblib.load("best_model.joblib")
except FileNotFoundError:
    st.error("Error: 'best_model.joblib' not found.")
    st.stop()

# ⚠ EXACT Training Columns (Required for the Safe Method)
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
    # Custom Title
    st.markdown('<div class="main-title">CerebroCare</div>', unsafe_allow_html=True)
    
    st.write("""
        ### AI-Powered Stroke Risk Assessment
        Welcome! This tool predicts the risk of stroke based on patient medical and lifestyle information.
        Fill out the form below to check the probability.
    """)
    st.divider()

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

    # =======================
    # PREDICT BUTTON
    # =======================
    if st.button("Analyze Risk Profile"):
        
        with st.spinner("Analyzing data..."):
            time.sleep(0.5) # Aesthetic loading delay
            
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

            # 2. Convert to DataFrame & Reindex (Safety Step)
            input_df = pd.DataFrame([input_dict])
            final_df = input_df.reindex(columns=MODEL_COLUMNS, fill_value=0)

            # 3. Predict
            prediction = model.predict(final_df)[0]
            probability = model.predict_proba(final_df)[0][1]

            st.subheader("Assessment Result")

            if prediction == 1:
                st.error(f"⚠ High Stroke Risk Detected\n\nProbability: {probability:.2%}")
                st.write("Please consult a medical professional immediately.")
            else:
                st.success(f"🟢 Low Stroke Risk Detected\n\nProbability: {probability:.2%}")
                st.write("Your metrics are within a safe range. Maintain a healthy lifestyle.")

if __name__ == "__main__":
    main()
