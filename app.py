import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go
from PIL import Image

# =======================
# 1. Configuration & CSS
# =======================
st.set_page_config(page_title="CerebroCare", layout="centered")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

        /* -- MAIN BODY -- */
        body {
            background-color: #0f172a;
            color: #e2e8f0;
            font-family: 'Roboto', sans-serif;
        }
        
        .stApp {
            background-color: #0f172a;
        }

        /* -- SIDEBAR STYLING -- */
        section[data-testid="stSidebar"] {
            background-color: #0f172a !important; 
            border-right: 1px solid #1e293b;
        }
        
        section[data-testid="stSidebar"] p, 
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div {
            color: #cbd5e1 !important;
        }

        section[data-testid="stSidebar"] img {
            margin-top: 20px;
            margin-bottom: 20px;
        }

        /* -- MAIN TITLES -- */
        .main-title {
            font-size: 3.5rem;
            font-weight: 700;
            color: #38bdf8;
            text-align: center;
            margin-bottom: -20px !important; 
            padding-bottom: 0px !important;
            font-family: 'Roboto', sans-serif;
        }
        
        /* -- TABS -- */
        button[data-baseweb="tab"] {
            color: #cbd5e1; 
            font-size: 20px !important;
            font-weight: 600;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #38bdf8;
            border-bottom-color: #38bdf8;
        }
        
        /* -- HEADERS -- */
        h3, .stHeader, .stSubheader {
            color: #f1f5f9 !important;
            font-size: 1.5rem !important;
            font-weight: 600 !important;
            margin-top: 25px !important;
            margin-bottom: 10px !important;
            background-color: transparent !important;
        }
        
        /* -- TEXT -- */
        p, li {
            color: #e2e8f0 !important;
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
            color: #94a3b8 !important;
        }

        /* Placeholder & Text Colors */
        input::placeholder {
            color: #94a3b8 !important;
            opacity: 1 !important; 
        }
        div[data-baseweb="select"] span {
            color: #94a3b8 !important;
        }
        div[data-baseweb="select"] div[aria-selected="true"] span {
            color: white !important;
        }
        div[data-baseweb="select"] svg {
            fill: #e2e8f0 !important;
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
    
    # --- SIDEBAR: LOGO & ABOUT ---
    with st.sidebar:
        try:
            col_l, col_center, col_r = st.columns([0.1, 0.8, 0.1])
            with col_center:
                img = Image.open("logo.jpeg")
                st.image(img, use_container_width=True)
        except Exception as e:
            st.warning("Logo not found.")
        
        st.markdown("---")
        
        st.markdown("""
            <div style='font-size: 12px; color: #cbd5e1; background-color: #1e293b; padding: 10px; border-radius: 5px; line-height: 1.4; border-left: 3px solid #38bdf8;'>
                <strong>Disclaimer:</strong> This tool provides risk estimation based on statistical models and should not replace professional medical diagnosis.
            </div>
        """, unsafe_allow_html=True)

    # --- MAIN CONTENT ---
    st.markdown('<div class="main-title">CerebroCare</div>', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0px !important; padding-top: 0px !important; text-align: center;'>AI-Powered Stroke Risk Assessment</h3>", unsafe_allow_html=True)

    # Init Session State
    if 'prediction_done' not in st.session_state:
        st.session_state['prediction_done'] = False
    if 'prediction_result' not in st.session_state:
        st.session_state['prediction_result'] = None
    if 'probability' not in st.session_state:
        st.session_state['probability'] = 0.0
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = {}

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🏠 Home Page", "🔍 Prediction", "📋 Personalized Result"])

    # ====================================================
    # TAB 1: HOME PAGE
    # ====================================================
    with tab1:
        st.markdown("<hr style='border: 1px solid #334155; margin-top: 0px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        
        st.subheader("What is a Stroke?")
        st.write("""
        A stroke occurs when the blood supply to part of your brain is interrupted or reduced, 
        preventing brain tissue from getting oxygen and nutrients. Brain cells begin to die in minutes. 
        **It is a medical emergency where immediate treatment is crucial.**
        """)
        
        st.markdown("<hr style='border: 1px solid #334155; margin: 15px 0;'>", unsafe_allow_html=True)

        st.subheader("Know the Warning Signs (:red[F.A.S.T.])")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("### 😐 :red[F]ace")
            st.caption("Does one side of the face droop or is it numb? Ask the person to smile.")
        with col2:
            st.markdown("### 💪 :red[A]rm")
            st.caption("Is one arm weak or numb? Ask the person to raise both arms.")
        with col3:
            st.markdown("### 🗣️ :red[S]peech")
            st.caption("Is speech slurred? Is the person unable to speak or hard to understand?")
        with col4:
            st.markdown("### 🚑 :red[T]ime")
            st.caption("If someone shows any of these signs, call emergency services immediately.")

        st.markdown("<hr style='border: 1px solid #334155; margin: 15px 0;'>", unsafe_allow_html=True)

        st.subheader("Risk Factors")
        risk_c1, risk_c2 = st.columns(2)
        with risk_c1:
            st.markdown("<h4 style='color: #cbd5e1; margin-bottom: 10px;'>🏥 Medical Risk Factors</h4>", unsafe_allow_html=True)
            st.markdown("""
            - High Blood Pressure (Hypertension)
            - High Cholesterol
            - Diabetes
            - Obesity
            - Family history of stroke
            """)
        with risk_c2:
            st.markdown("<h4 style='color: #cbd5e1; margin-bottom: 10px;'>🚬 Lifestyle Risk Factors</h4>", unsafe_allow_html=True)
            st.markdown("""
            - Smoking or tobacco use
            - Physical inactivity
            - Heavy alcohol consumption
            - Unhealthy diet (high salt & saturated fats)
            """)

        st.markdown("<hr style='border: 1px solid #334155; margin: 15px 0;'>", unsafe_allow_html=True)

        st.subheader("Time is Brain 🧠")
        st.info("""
        The faster a stroke is treated, the more likely the patient is to recover. 
        Early detection of risk factors through AI analysis can help prevent a stroke before it happens 
        by enabling timely lifestyle changes and medical intervention.
        """)

        st.subheader("Lower Your Risk Today")
        st.success("""
        - **Control Blood Pressure:** Keep it in a healthy range.
        - **Stay Active:** Aim for at least 30 minutes of moderate exercise daily.
        - **Eat Healthy:** Focus on fruits, vegetables, and whole grains; limit salt.
        - **Quit Smoking:** Smoking thickens your blood and increases plaque buildup in arteries.
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; color: #94a3b8;'>👇 <i>Go to the <b>Prediction</b> tab to check your profile</i> 👇</div>", unsafe_allow_html=True)

    # ====================================================
    # TAB 2: PREDICTION
    # ====================================================
    with tab2:
        st.markdown("<hr style='border: 1px solid #334155; margin-top: 0px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        
        st.subheader("Patient Info")
        age = st.number_input("Age", min_value=0, max_value=120, value=None, placeholder="e.g. 45")
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

        if st.button("Analyze Stroke Risk"):
            required_fields = [age, gender, ever_married, residence, bmi, hypertension, heart_disease, smoking_display, work_display, avg_glucose_level]
            
            if None in required_fields:
                st.error("⚠ Please fill out all fields before analyzing.")
            else:
                with st.spinner("Analyzing data..."):
                    time.sleep(0.5) 
                    
                    raw_work_type = WORK_TYPE_MAP[work_display]
                    raw_smoking_status = SMOKING_MAP[smoking_display]

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

    # ====================================================
    # TAB 3: PERSONALIZED RESULT
    # ====================================================
    with tab3:
        st.markdown("<hr style='border: 1px solid #334155; margin-top: 0px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        st.subheader("Personalized Insights")
        
        if st.session_state['prediction_done']:
            prob = st.session_state['probability']
            user_data = st.session_state['user_input']
            
            # Gauge Chart
            col_graph1, col_graph2 = st.columns([1, 1])
            with col_graph1:
                st.markdown("**Risk Probability Gauge**")
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
                        'threshold': {
                            'line': {'color': "white", 'width': 5},
                            'thickness': 0.8,
                            'value': prob * 100
                        }
                    }
                ))
                fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=300, margin=dict(l=30, r=30, t=50, b=30))
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Bar Chart
            with col_graph2:
                st.markdown("**Your Metrics vs Healthy Average**")
                categories = ['BMI', 'Glucose', 'Age']
                user_values = [user_data['BMI'], user_data['Glucose'], user_data['Age']]
                healthy_values = [22.0, 90.0, 40.0]
                
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
                    legend=dict(
                        orientation="h", 
                        yanchor="bottom", y=1.02, 
                        xanchor="right", x=1,
                        font=dict(color="white")
                    )
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("---")
            st.write("### AI-Generated Recommendations")
            
            # --- TIERED RECOMMENDATION LOGIC ---
            
            # 1. LOW RISK (0 - 40%)
            if prob < 0.4:
                st.success("✅ **Result: Low Risk – Optimal Health Maintenance**")
                st.markdown("**Overview:**")
                st.write("Great news! Your profile indicates a healthy cardiovascular system with minimal risk factors. While your current risk is low, your goal now is consistency. Stroke prevention is a lifelong commitment, not a one-time achievement.")
                
                st.markdown("**Action Plan:**")
                st.markdown("""
                - **🥗 Nutrition:** Focus on "preventative nutrition." Maintain a diet rich in antioxidants, fiber, and Omega-3 fatty acids (like salmon or walnuts) to protect blood vessels.
                - **🏃 Physical Maintenance:** Keep your body moving. Even if you are busy, ensure you get at least 30 minutes of walking or light activity daily to keep blood circulation flowing smoothly.
                - **🧘 Mental Well-being:** Stress can silently raise blood pressure. Continue practicing stress-management techniques like deep breathing, meditation, or simply taking time for hobbies.
                - **📅 Routine Monitoring:** Even with low risk, don’t skip your annual medical check-up. Track your blood pressure once every few months to ensure it remains stable.
                """)

            # 2. MEDIUM RISK (40% - 70%)
            elif prob < 0.7:
                st.warning("⚠️ **Result: Medium Risk – Warning Signs Detected**")
                st.markdown("**Overview:**")
                st.write("Your analysis shows several indicators that need attention. You are in the \"warning zone.\" This is actually a good opportunity—by acting now, you can reverse these risk factors before they become severe issues.")
                
                st.markdown("**Action Plan:**")
                st.markdown("""
                - **🥑 Targeted Diet Change:** It’s time to get specific. Cut down on processed foods, sugary drinks, and excessive red meat. Shift towards the Mediterranean Diet (rich in olive oil, veggies, and fish) which is proven to support heart health.
                - **⚖️ Weight Management:** If your BMI is above average, losing even 5-10% of your body weight can significantly lower your stroke risk and blood pressure.
                - **🚴 Active Lifestyle:** Move beyond just "walking." Try to incorporate moderate cardio that raises your heart rate (like jogging, swimming, or cycling) at least 3-4 times a week.
                - **📝 Habit Audit:** Identify one major bad habit (e.g., occasional smoking, late-night snacking, or sedentary weekends) and commit to eliminating it this month.
                """)

            # 3. HIGH RISK (> 70%)
            else:
                st.error("🚨 **Result: High Risk – Critical Action Required**")
                st.markdown("**Overview:**")
                st.write("This result indicates that multiple strong risk factors are present. This is a serious alert. Your cardiovascular system is under stress, and without immediate changes, the likelihood of a stroke or heart event is significant.")
                
                st.markdown("**Action Plan:**")
                st.markdown("""
                - **👨‍⚕️ Medical Intervention:** **Do not ignore this.** Schedule an appointment with a cardiologist or general practitioner this week. Share this risk assessment with them to discuss medication or treatment plans.
                - **📉 Strict Monitoring:** Monitor your blood pressure daily. Keep a log of your numbers to show your doctor. If your BP exceeds 140/90 consistently, seek medical help.
                - **🆘 Emergency Plan:** Educate yourself and your family members about the **F.A.S.T.** signs (Face drooping, Arm weakness, Speech difficulty). Ensure everyone knows what to do in an emergency.
                - **🛑 Radical Lifestyle Reset:**
                    - **Zero Tolerance:** If you smoke, stop immediately.
                    - **Diet:** Switch to a strict low-sodium, heart-healthy diet immediately.
                    - **Support System:** Tell a family member or friend about this result so they can support you in making these difficult but necessary lifestyle changes.
                """)

            # DISCLAIMER DI BAWAH REKOMENDASI
            st.markdown("---")
            st.caption("This AI-powered assessment provides an estimation based on your inputs and implies statistical probability, not a definitive medical prediction. Please consult a healthcare professional for clinical diagnosis.")

        else:
            st.markdown("""
                <div style='background-color: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155; text-align: center;'>
                    <h4 style='color: #cbd5e1;'>No Analysis Data Found</h4>
                    <p style='color: #94a3b8;'>Please go to the <b>Prediction</b> tab and fill out the form to generate personalized results.</p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
