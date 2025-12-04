import streamlit as st
import pandas as pd
import joblib
import time

# =======================
# 1. Configuration & CSS (Green/Olive Theme Fixed)
# =======================
st.set_page_config(page_title="Stroke Risk Prediction", layout="centered")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Playfair+Display:wght@600;700&display=swap');

        /* -- GLOBAL TEXT & BACKGROUND -- */
        body {
            background-color: #2C3930;
            color: #DCD7C9;
            font-family: 'Poppins', sans-serif;
        }

        .stApp {
            background-color: #2C3930;
        }

        /* -- TITLES & HEADERS -- */
        h1, h2, h3, .stTitle, .stHeader, .stSubheader {
            color: #DCD7C9 !important;
            font-family: 'Playfair Display', serif;
        }

        /* -- WIDGET LABELS (Age, Gender, etc.) -- */
        /* UPDATE: Selector ini diperbaiki agar label benar-benar terlihat */
        .stNumberInput label, 
        .stSelectbox label, 
        .stTextInput label,
        div[data-testid="stWidgetLabel"] p {
            color: #DCD7C9 !important;
            font-size: 16px !important;
            font-weight: 600 !important;
        }

        /* -- INPUT BOXES & DROPDOWNS -- */
        div[data-baseweb="select"] > div, 
        div[data-baseweb="input"] > div {
            background-color: #3F4F44 !important;
            color: #DCD7C9 !important;
            border: 1px solid #6E8E59 !important;
            border-radius: 10px;
        }

        /* Warna teks input angka dan pilihan dropdown */
        input[type="number"], div[data-baseweb="select"] span {
            color: #DCD7C9 !important;
        }
        
        /* Warna ikon panah dropdown */
        div[data-baseweb="select"] svg {
            fill: #DCD7C9 !important;
        }

        /* -- BUTTON STYLING -- */
        .stButton>button {
            background-color: #6E8E59;
            color: #DCD7C9;
            font-size: 18px;
            padding: 12px 20px;
            border-radius: 10px;
            border: none;
            font-weight: 600;
            transition: 0.3s;
            width: 100%;
            font-family: 'Poppins', sans-serif;
        }

        .stButton>button:hover {
            background-color: #5c7749;
            color: #fff;
        }

        /* -- SUCCESS/ERROR MESSAGE BOX -- */
        .stAlert {
            background-color: #3F4F44;
            color: #DCD7C9;
            border: 1px solid #6E8E59;
        }
    </style>
""", unsafe_allow_html=True)


# =======================
# 2. Load Trained Model
# =======================
try:
    model = joblib.load("best_model.joblib")
except FileNotFoundError:
    st.error("Error: 'best_model.joblib' not found. Please upload the model file.")
    st.stop()

# Daftar kolom harus SAMA PERSIS dengan data training
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
    st.title("🧠 Stroke Risk Prediction App")

    st.write("""
        Welcome!  
        This tool predicts the *risk of stroke* based on patient medical and lifestyle information.  
        Fill out the form below to check the probability.
    """)
    st.markdown("---")

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

    st.markdown("<br>", unsafe_allow_html=True)

    # =======================
    # PREDICT BUTTON
    # =======================
    # Text tombol sudah diubah sesuai permintaan
    if st.button("Analyze Stroke Risk"):
        
        # 1. Menyiapkan Data Input
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

        # 2. Konversi ke DataFrame & Sesuaikan Kolom (Agar aman dari error)
        input_df = pd.DataFrame([input_dict])
        final_df = input_df.reindex(columns=MODEL_COLUMNS, fill_value=0)

        # 3. Prediksi
        with st.spinner("Analyzing data..."):
            time.sleep(0.5) # Efek visual loading
            prediction = model.predict(final_df)[0]
            probability = model.predict_proba(final_df)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"⚠ High Stroke Risk\n\nProbability: {probability:.2%}")
        else:
            st.success(f"🟢 Low Stroke Risk\n\nProbability: {probability:.2%}")

if __name__ == "__main__":
    main()
