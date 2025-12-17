import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go

# =======================
# 1. Configuration & CSS
# =======================
st.set_page_config(page_title="CerebroCare", layout="centered")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

        body {
            background-color: #0f172a;
            color: #e2e8f0;
            font-family: 'Roboto', sans-serif;
        }
        
        .stApp {
            background-color: #0f172a;
        }

        .main-title {
            font-size: 3.5rem;
            font-weight: 700;
            color: #38bdf8;
            text-align: center;
            margin-bottom: 5px;
            font-family: 'Roboto', sans-serif;
        }
        
        button[data-baseweb="tab"] {
            color: #cbd5e1; 
            font-size: 16px;
            font-weight: 600;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #38bdf8;
            border-bottom-color: #38bdf8;
        }
        
        h3, .stHeader, .stSubheader {
            color: #f1f5f9 !important;
            font-size: 1.5rem !important;
            font-weight: 600 !important;
            margin-top: 25px !important;
            margin-bottom: 10px !important;
            background-color: transparent !important;
        }
        
        /* REVISI 3: Menambahkan 'li' agar bullet points menjadi terang */
        p, li {
            color: #e2e8f0 !important;
            font-size: 16px;
            line-height: 1.6;
        }

        /* Label Widget (Age, Gender, dll) - WARNA PUTIH TERANG */
        .stNumberInput label, 
        .stSelectbox label, 
        .stTextInput label,
        div[data-testid="stWidgetLabel"] p {
            color: #f1f5f9 !important;
            font-size: 16px !important;
            margin-bottom: 5px;
        }

        /* ----------------------------------------------------
           CUSTOM INPUT BOX STYLING
        ---------------------------------------------------- */
        
        /* Reset background container */
        .stSelectbox, .stNumberInput, .stTextInput {
            background-color: transparent !important;
            border: none !important;
        }

        /* 1. Box Dropdown & Input */
        div[data-baseweb="select"] > div, 
        div[data-baseweb="input"] {
            background-color: #1e293b !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
            color: white !important; /* Warna teks yang sudah diketik/dipilih */
        }

        /* 2. REVISI: Placeholder Text (e.g. 45, Select Gender) 
           Warna Abu Terang (#94a3b8) - Terlihat jelas tapi beda dgn Label */
        
        /* Untuk Input Angka (NumberInput) */
        input::placeholder {
            color: #94a3b8 !important;
            opacity: 1 !important; 
        }
        
        /* Untuk Dropdown (Selectbox) - Placeholder text */
        div[data-baseweb="select"] span {
            color: #94a3b8 !important; /* Warna default (placeholder) */
        }
        
        /* Memastikan nilai yang DIPILIH di dropdown berwarna PUTIH (bukan abu) */
        div[data-baseweb="select"] div[aria-selected="true"] span {
            color: white !important;
        }

        /* 3. REVISI: Warna Panah Dropdown (Arrow)
           Disamakan dengan warna ikon +/- yaitu Putih/Terang */
        div[data-baseweb="select"] svg {
            fill: #e2e8f0 !important;
        }

        /* Styling internal input box */
        div[data-baseweb="input"] > div {
            background-color: transparent !important;
            color: white !important;
        }
        
        input[class] {
            color: white !important;
            background-color: transparent !important;
        }

        /* Dropdown Popover (Pilihan Menu) */
        div[data-baseweb="popover"] div {
            background-color: #1e293b !important;
            color: white !important;
        }
        
        /* Tombol Analyze */
        .stButton>button {
            background-color: #0ea5e9;
            color: white;
            font-size: 18px;
            padding: 12px 20px;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            transition: 0.3s;
            width: 100%;
            margin-top: 30px;
        }
        .stButton>button:hover {
            background-color: #0284c7;
        }

        .stAlert {
            background-color: #1e293b;
            color: white;
            border: 1px solid #334155;
        }
    </style>
""", unsafe_allow_html=True)


# =======================
# 2. Load Model & Prepare Mappings
# =======================
try:
    model = joblib.load("best_model.joblib")
except FileNotFoundError:
    st.error("Error: 'best_model.joblib' not found.")
    st.stop()

# ⚠ EXACT Training Columns
MODEL_COLUMNS = [
    "age", "hypertension", "heart_disease", "ever_married", "avg_glucose_level", 
    "bmi", "gender_Male", "work_type_Never_worked", "work_type_Private", 
    "work_type_Self-employed", "work_type_children", "Residence_type_Urban", 
    "smoking_status_formerly smoked", "smoking_status_never smoked", "smoking_status_smokes"
]

# --- MAPPING: Tampilan Rapi -> Data Asli Model ---
WORK_TYPE_MAP = {
    "Private Sector": "Private",
    "Self Employed": "Self-employed",
    "Government Job": "Govt_job",
    "Never Worked": "Never_worked",
    "Children": "children"
}

SMOKING_MAP = {
    "Never Smoked": "never smoked",
    "Formerly Smoked": "formerly smoked",
    "Currently Smokes": "smokes",
    "Unknown": "Unknown"
}

# =======================
# 3. MAIN APP
# =======================
def main():
    st.markdown('<div class="main-title">CerebroCare</div>', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0px !important; text-align: center;'>AI-Powered Stroke Risk Assessment</h3>", unsafe_allow_html=True)

    if 'prediction_done' not in st.session_state:
        st.session_state['prediction_done'] = False
    if 'prediction_result' not in st.session_state:
        st.session_state['prediction_result'] = None
    if 'probability' not in st.session_state:
        st.session_state['probability'] = 0.0
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = {}

    # ==========================
    # TABS
    # ==========================
    tab1, tab2, tab3 = st.tabs(["🏠 Home Page", "🔍 Prediction", "📋 Personalized Result"])

    # ----------------------------------------------------
    # TAB 1: HOME PAGE
    # ----------------------------------------------------
    with tab1:
        st.markdown("<hr style='border: 1px solid #334155; margin-top: 0px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        st.subheader("About Stroke")
        st.write("""
        A stroke occurs when blood supply to part of the brain is blocked or a blood vessel bursts. 
        It can cause lasting brain damage, long-term disability, or even death.
        """)
        st.subheader("Why Early Detection Matters?")
        st.write("""
        Risk factors like hypertension, heart disease, and lifestyle choices can be managed. 
        **CerebroCare** analyzes these factors to estimate your risk profile.
        """)
        st.info("Navigate to the **Prediction** tab to start your assessment.")

    # ----------------------------------------------------
    # TAB 2: PREDICTION
    # ----------------------------------------------------
    with tab2:
        st.markdown("<hr style='border: 1px solid #334155; margin-top: 0px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        
        # --- INPUT FORM (DEFAULT EMPTY) ---
        st.subheader("Patient Info")
        
        # NOTE: value=None makes it empty, placeholder shows text hint
        age = st.number_input("Age", min_value=0, max_value=120, value=None, placeholder="e.g. 45")
        
        # NOTE: index=None makes dropdown empty/unselected
        gender = st.selectbox("Gender", ["Male", "Female"], index=None, placeholder="Select Gender")
        ever_married = st.selectbox("Ever Married?", ["Yes", "No"], index=None, placeholder="Select Status")
        residence = st.selectbox("Residence Type", ["Urban", "Rural"], index=None, placeholder="Select Residence Type")
        bmi = st.number_input("BMI", min_value=0.0, value=None, placeholder="e.g. 24.5")

        st.subheader("Medical History")
        hypertension = st.selectbox("Hypertension", ["Yes", "No"], index=None, placeholder="Select History")
        heart_disease = st.selectbox("Heart Disease", ["Yes", "No"], index=None, placeholder="Select History")
        
        smoking_display = st.selectbox("Smoking Status", list(SMOKING_MAP.keys()), index=None, placeholder="Select Smoking Status")
        work_display = st.selectbox("Work Type", list(WORK_TYPE_MAP.keys()), index=None, placeholder="Select Work Type")
        
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=None, placeholder="e.g. 95.0")

        # --- PREDICT BUTTON ---
        if st.button("Analyze Stroke Risk"):
            
            # 1. VALIDATION CHECK (PENTING!)
            # Cek apakah ada input yang masih kosong (None)
            required_fields = [age, gender, ever_married, residence, bmi, hypertension, heart_disease, smoking_display, work_display, avg_glucose_level]
            
            if None in required_fields:
                st.error("⚠ Please fill out all fields before analyzing.")
            else:
                # Jika semua terisi, baru jalankan proses prediksi
                with st.spinner("Analyzing data..."):
                    time.sleep(0.5) 
                    
                    # Konversi Pilihan Tampilan -> Nilai Asli Dataset
                    raw_work_type = WORK_TYPE_MAP[work_display]
                    raw_smoking_status = SMOKING_MAP[smoking_display]

                    # Create Input Dictionary
                    input_dict = {
                        "age": age,
                        "hypertension": 1 if hypertension == "Yes" else 0,
                        "heart_disease": 1 if heart_disease == "Yes" else 0,
                        "ever_married": 1 if ever_married == "Yes" else 0,
                        "avg_glucose_level": avg_glucose_level,
                        "bmi": bmi,
                        "gender_Male": 1 if gender == "Male" else 0,
                        
                        "work_type_Never_worked": 1 if raw_work_type == "Never_worked" else 0,
                        "work_type_Private": 1 if raw_work_type == "Private" else 0,
                        "work_type_Self-employed": 1 if raw_work_type == "Self-employed" else 0,
                        "work_type_children": 1 if raw_work_type == "children" else 0,
                        
                        "Residence_type_Urban": 1 if residence == "Urban" else 0,
                        
                        "smoking_status_formerly smoked": 1 if raw_smoking_status == "formerly smoked" else 0,
                        "smoking_status_never smoked": 1 if raw_smoking_status == "never smoked" else 0,
                        "smoking_status_smokes": 1 if raw_smoking_status == "smokes" else 0,
                    }

                    input_df = pd.DataFrame([input_dict])
                    final_df = input_df.reindex(columns=MODEL_COLUMNS, fill_value=0)
                    
                    prediction = model.predict(final_df)[0]
                    probability = model.predict_proba(final_df)[0][1]

                    st.session_state['prediction_done'] = True
                    st.session_state['prediction_result'] = prediction
                    st.session_state['probability'] = probability
                    
                    st.session_state['user_input'] = {
                        "Age": age,
                        "BMI": bmi,
                        "Glucose": avg_glucose_level
                    }
        
        # --- DISPLAY RESULT ---
        if st.session_state['prediction_done']:
            st.markdown("---")
            st.subheader("Prediction Result")
            pred = st.session_state['prediction_result']
            prob = st.session_state['probability']

            if pred == 1:
                st.error(f"⚠ High Stroke Risk Detected\n\nProbability: {prob:.2%}")
                st.write("Please consult a medical professional immediately.")
            else:
                st.success(f"🟢 Low Stroke Risk Detected\n\nProbability: {prob:.2%}")
                st.write("Your metrics are within a safe range. Maintain a healthy lifestyle.")
                
            st.info("👉 Check the **Personalized Result** tab for Visual Analytics.")

    # ----------------------------------------------------
    # TAB 3: PERSONALIZED RESULT
    # ----------------------------------------------------
    with tab3:
        st.markdown("<hr style='border: 1px solid #334155; margin-top: 0px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        st.subheader("Personalized Insights")
        
        if st.session_state['prediction_done']:
            prob = st.session_state['probability']
            user_data = st.session_state['user_input']
            
            # GRAFIK 1: GAUGE
            col_graph1, col_graph2 = st.columns([1, 1])
            with col_graph1:
                st.markdown("**Risk Probability Gauge**")
                
                # REVISI 1: Menambahkan Threshold Line (Jarum)
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    title = {'text': "Stroke Probability (%)", 'font': {'color': 'white'}},
                    number = {'font': {'color': 'white'}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#0ea5e9"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': "#10b981"},
                            {'range': [40, 70], 'color': "#f59e0b"},
                            {'range': [70, 100], 'color': "#ef4444"}
                        ],
                        # JARUM INDIKATOR
                        'threshold': {
                            'line': {'color': "white", 'width': 5},
                            'thickness': 0.8,
                            'value': prob * 100
                        }
                    }
                ))
                fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=300, margin=dict(l=30, r=30, t=50, b=30))
                st.plotly_chart(fig_gauge, use_container_width=True)

            # GRAFIK 2: BAR CHART
            with col_graph2:
                st.markdown("**Your Metrics vs Healthy Average**")
                categories = ['BMI', 'Glucose', 'Age']
                user_values = [user_data['BMI'], user_data['Glucose'], user_data['Age']]
                healthy_values = [22.0, 90.0, 40.0]
                
                fig_bar = go.Figure(data=[
                    go.Bar(name='Your Data', x=categories, y=user_values, marker_color='#38bdf8'),
                    go.Bar(name='Healthy Avg', x=categories, y=healthy_values, marker_color='#94a3b8')
                ])
                
                # REVISI 2: Menambahkan warna font Legend menjadi Putih
                fig_bar.update_layout(
                    barmode='group', 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    font={'color': "white"}, 
                    height=300, 
                    margin=dict(l=20, r=20, t=50, b=20), 
                    legend=dict(
                        orientation="h", 
                        yanchor="bottom", y=1.02, 
                        xanchor="right", x=1,
                        font=dict(color="white") # Legend Putih
                    )
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("---")
            st.write("### AI-Generated Recommendations")
            
            # REVISI 3: Tulisan (Bullet Points) otomatis terang karena CSS "li {color: #e2e8f0}"
            if prob > 0.5:
                st.warning("Based on the analysis, your risk profile is elevated.")
                st.markdown("""
                - **Primary Concern:** Your calculated probability is in the higher range.
                - **Comparison:** Check the bar chart. If your Glucose or BMI is significantly higher than the 'Healthy Avg', focus on lowering that metric.
                - **Action:** Consult a specialist immediately.
                """)
            else:
                st.success("Your risk profile is currently stable.")
                st.markdown("""
                - **Comparison:** Your metrics align well with healthy averages.
                - **Maintenance:** Continue your current diet and exercise routine.
                """)
        else:
            st.markdown(
                """
                <div style='background-color: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155; text-align: center;'>
                    <h4 style='color: #cbd5e1;'>No Analysis Data Found</h4>
                    <p style='color: #94a3b8;'>Please go to the <b>Prediction</b> tab and fill out the form to generate personalized results.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()
