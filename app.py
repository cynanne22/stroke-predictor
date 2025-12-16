import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go # PERLU: Import library untuk grafik

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
            margin-bottom: 5px;
            font-family: 'Roboto', sans-serif;
        }
        
        /* -- TABS STYLING -- */
        button[data-baseweb="tab"] {
            color: #cbd5e1; 
            font-size: 16px;
            font-weight: 600;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #38bdf8; /* Cyan */
            border-bottom-color: #38bdf8;
        }
        
        /* -- SUBHEADERS -- */
        h3, .stHeader, .stSubheader {
            color: #f1f5f9 !important;
            font-size: 1.5rem !important;
            font-weight: 600 !important;
            margin-top: 25px !important;
            margin-bottom: 10px !important;
            background-color: transparent !important;
        }
        
        p {
            color: #e2e8f0;
            font-size: 16px;
            line-height: 1.6;
        }

        /* -- WIDGET LABELS -- */
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
        .stSelectbox, .stNumberInput, .stTextInput {
            background-color: transparent !important;
            border: none !important;
        }

        div[data-baseweb="select"] > div, 
        div[data-baseweb="input"] {
            background-color: #1e293b !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
            color: white !important;
        }

        div[data-baseweb="input"] > div {
            background-color: transparent !important;
            color: white !important;
        }
        
        input[class] {
            color: white !important;
            background-color: transparent !important;
        }

        div[data-baseweb="popover"] div {
            background-color: #1e293b !important;
            color: white !important;
        }
        div[data-baseweb="select"] span {
            color: white !important;
        }

        /* -- BUTTONS -- */
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

# ⚠ EXACT Training Columns
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
    st.markdown('<div class="main-title">CerebroCare</div>', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0px !important; text-align: center;'>AI-Powered Stroke Risk Assessment</h3>", unsafe_allow_html=True)

    # Inisialisasi Session State
    if 'prediction_done' not in st.session_state:
        st.session_state['prediction_done'] = False
    if 'prediction_result' not in st.session_state:
        st.session_state['prediction_result'] = None
    if 'probability' not in st.session_state:
        st.session_state['probability'] = 0.0
    
    # Simpan input user juga agar bisa divisualisasikan di Tab 3
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = {}

    # ==========================
    # TABS IMPLEMENTATION
    # ==========================
    tab1, tab2, tab3 = st.tabs(["🏠 Home Page", "🔍 Prediction", "📋 Personalized Result"])

    # ----------------------------------------------------
    # TAB 1: HOME PAGE
    # ----------------------------------------------------
    with tab1:
        st.markdown("<hr style='border: 1px solid #334155; margin-top: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        
        st.subheader("About Stroke")
        st.write("""
        A stroke, sometimes called a brain attack, occurs when something blocks blood supply to part of the brain or when a blood vessel in the brain bursts. 
        In either case, parts of the brain become damaged or die. A stroke can cause lasting brain damage, long-term disability, or even death.
        """)

        st.subheader("Why Early Detection Matters?")
        st.write("""
        Risk factors such as high blood pressure, heart disease, smoking, and diabetes can be managed if detected early. 
        **CerebroCare** uses Artificial Intelligence to analyze your health metrics and estimate your potential risk profile.
        """)
        
        st.info("Navigate to the **Prediction** tab to start your assessment.")

    # ----------------------------------------------------
    # TAB 2: PREDICTION
    # ----------------------------------------------------
    with tab2:
        st.markdown(
            """
            <p style='color: #cbd5e1; font-size: 16px; margin-top: 10px; margin-bottom: 10px; text-align: center;'>
            Fill out the patient details below to analyze the risk profile.
            </p>
            """, 
            unsafe_allow_html=True
        )
        st.markdown("<hr style='border: 1px solid #334155; margin-top: 0px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        
        # --- INPUT FORM ---
        st.subheader("Patient Info")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        ever_married = st.selectbox("Ever Married?", ["Yes", "No"])
        residence = st.selectbox("Residence Type", ["Urban", "Rural"])
        bmi = st.number_input("BMI", min_value=0.0, value=25.0)

        st.subheader("Medical History")
        hypertension = st.selectbox("Hypertension", ["Yes", "No"])
        heart_disease = st.selectbox("Heart Disease", ["Yes", "No"])
        smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Never_worked", "children", "Govt_job"])
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=90.0)

        # --- PREDICT BUTTON ---
        if st.button("Analyze Stroke Risk"):
            with st.spinner("Analyzing data..."):
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

                input_df = pd.DataFrame([input_dict])
                final_df = input_df.reindex(columns=MODEL_COLUMNS, fill_value=0)
                
                prediction = model.predict(final_df)[0]
                probability = model.predict_proba(final_df)[0][1]

                # SAVE TO SESSION STATE
                st.session_state['prediction_done'] = True
                st.session_state['prediction_result'] = prediction
                st.session_state['probability'] = probability
                
                # Simpan data mentah user utk visualisasi
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
                st.error(f"⚠️ High Stroke Risk Detected\n\nProbability: {prob:.2%}")
                st.write("Please consult a medical professional immediately.")
            else:
                st.success(f"🟢 Low Stroke Risk Detected\n\nProbability: {prob:.2%}")
                st.write("Your metrics are within a safe range. Maintain a healthy lifestyle.")
                
            st.info("👉 Check the **Personalized Result** tab for Visual Analytics.")

    # ----------------------------------------------------
    # TAB 3: PERSONALIZED RESULT (WITH CHARTS)
    # ----------------------------------------------------
    with tab3:
        st.subheader("Personalized Insights")
        
        if st.session_state['prediction_done']:
            
            # 1. Ambil Data
            prob = st.session_state['probability']
            user_data = st.session_state['user_input']
            
            # -----------------------------------------
            # GRAFIK 1: GAUGE CHART (SPIDOMETER RISIKO)
            # -----------------------------------------
            col_graph1, col_graph2 = st.columns([1, 1])
            
            with col_graph1:
                st.markdown("**Risk Probability Gauge**")
                
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100, # Convert to percentage
                    title = {'text': "Stroke Probability (%)", 'font': {'color': 'white'}},
                    number = {'font': {'color': 'white'}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#0ea5e9"}, # Warna Jarum/Bar (Biru Langit)
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': "#10b981"},  # Green (Safe)
                            {'range': [40, 70], 'color': "#f59e0b"}, # Yellow (Warning)
                            {'range': [70, 100], 'color': "#ef4444"} # Red (Danger)
                        ],
                    }
                ))
                
                # Update layout agar transparan menyatu dengan background
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "white"},
                    height=300,
                    margin=dict(l=30, r=30, t=50, b=30)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            # -----------------------------------------
            # GRAFIK 2: COMPARISON CHART (USER VS HEALTHY)
            # -----------------------------------------
            with col_graph2:
                st.markdown("**Your Metrics vs Healthy Average**")
                
                # Standar kesehatan kasar untuk perbandingan
                categories = ['BMI', 'Glucose', 'Age']
                user_values = [user_data['BMI'], user_data['Glucose'], user_data['Age']]
                healthy_values = [22.0, 90.0, 40.0] # Contoh nilai rata-rata sehat
                
                fig_bar = go.Figure(data=[
                    go.Bar(name='Your Data', x=categories, y=user_values, marker_color='#38bdf8'),
                    go.Bar(name='Healthy Avg', x=categories, y=healthy_values, marker_color='#94a3b8')
                ])
                
                fig_bar.update_layout(
                    barmode='group',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "white"},
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # --- Text Advice ---
            st.markdown("---")
            st.write("### AI-Generated Recommendations")
            
            if prob > 0.5:
                st.warning("Based on the analysis, your risk profile is elevated.")
                st.markdown("""
                - **Primary Concern:** Your calculated probability is in the higher range.
                - **Comparison:** Check the bar chart. If your Glucose or BMI is significantly higher than the 'Healthy Avg' (Grey bar), focus on lowering that metric.
                - **Action:** Consult a specialist immediately.
                """)
            else:
                st.success("Your risk profile is currently stable.")
                st.markdown("""
                - **Comparison:** Your metrics align well with healthy averages.
                - **Maintenance:** Continue your current diet and exercise routine.
                """)
            
        else:
            # Tampilan jika belum melakukan analisis
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
