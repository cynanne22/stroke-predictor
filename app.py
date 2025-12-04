import streamlit as st

import pandas as pd

import joblib

import time

import os

from pathlib import Path



# =======================

# 1. Page Configuration

# =======================

st.set_page_config(

    page_title="CerebroCare",

    page_icon="🧠",

    layout="wide",

    initial_sidebar_state="expanded"

)



# =======================

# 2. Custom CSS Styling (Enhanced Medical Navy Theme)

# =======================

st.markdown("""

    <style>

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');



        * {

            font-family: 'Inter', sans-serif;

        }



        /* -- MAIN BODY COLORS -- */

        body {

            background-color: #0f172a;

            color: #e2e8f0;

        }

        

        .stApp {

            background-color: #0f172a;

        }



        /* -- SIDEBAR STYLING -- */

        .stSidebar {

            background-color: #1a2847;

            border-right: 2px solid #0ea5e9;

        }



        .stSidebar .stSelectbox label {

            color: #38bdf8 !important;

            font-weight: 600;

            font-size: 14px !important;

        }



        /* -- MAIN TITLE -- */

        h1, h2 {

            color: #38bdf8 !important;

            font-weight: 700;

        }



        h3, h4, h5, h6 {

            color: #f1f5f9 !important;

            font-weight: 600;

        }



        p, .stMarkdown {

            color: #cbd5e1;

            font-size: 15px;

            line-height: 1.6;

        }



        /* -- WIDGET LABELS -- */

        .stNumberInput label,

        .stSelectbox label,

        .stTextInput label,

        div[data-tested="stWidgetLabel"] p {

            color: #cbd5e1 !important;

            font-weight: 500;

            font-size: 14px !important;

        }



        /* -- INPUT FIELDS -- */

        .stNumberInput > div > input,

        .stTextInput > div > input,

        .stSelectbox > div > div {

            background-color: #1e293b !important;

            color: #f1f5f9 !important;

            border: 1.5px solid #334155 !important;

            border-radius: 8px !important;

            padding: 10px 12px !important;

            font-size: 15px !important;

        }



        .stNumberInput > div > input:focus,

        .stTextInput > div > input:focus {

            border-color: #0ea5e9 !important;

            background-color: #1e293b !important;

        }



        /* -- DROPDOWN -- */

        div[data-baseweb="select"] > div {

            background-color: #1e293b !important;

            border-color: #334155 !important;

        }



        div[data-baseweb="select"] span {

            color: #f1f5f9 !important;

        }



        div[data-baseweb="popover"] {

            background-color: #1e293b !important;

        }



        /* -- BUTTONS -- */

        .stButton > button {

            background-color: #0ea5e9;

            color: white;

            font-weight: 600;

            font-size: 16px;

            padding: 12px 24px !important;

            border-radius: 8px;

            border: none;

            transition: all 0.3s ease;

            width: 100%;

        }



        .stButton > button:hover {

            background-color: #0284c7;

            transform: translate(-2px);

            box-shadow: 0 8px 16px rgba(14, 165, 233, 0.3);

        }



        .stButton > button:active {

            background-color: #0369a1;

        }



        /* -- ALERTS/MESSAGES -- */

        .stAlert {

            background-color: #1e293b;

            color: #f1f5f9;

            border-left: 4px solid #0ea5e9;

            border-radius: 8px;

            padding: 16px;

        }



        /* Success Alert */

        .stAlert[data-tested="stSuccess"] {

            border-left-color: #10b981;

        }



        /* Error Alert */

        .stAlert[data-tested="stError"] {

            border-left-color: #ef4444;

        }



        /* -- DIVIDER -- */

        .stDivider {

            border-color: #334155;

        }



        /* -- METRIC CARDS -- */

        .metric-card {

            background-color: #1e293b;

            border: 1px solid #334155;

            border-radius: 12px;

            padding: 20px;

            margin: 10px 0;

            transition: all 0.3s ease;

        }



        .metric-card:hover {

            border-color: #0ea5e9;

            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.2);

        }



        .metric-label {

            color: #94a3b8;

            font-size: 13px;

            font-weight: 500;

            text-transform: uppercase;

            letter-spacing: 0.5px;

        }



        .metric-value {

            color: #38bdf8;

            font-size: 32px;

            font-weight: 700;

            margin: 8px 0 0 0;

        }

    </style>

""", unsafe_allow_html=True)



# =======================

# 3. Load Model (Safe)

# =======================

@st.cache_resource

def load_model():

    try:

        model_path = "best_model.joblib"

        if os.path.exists(model_path):

            return joblib.load(model_path)

        else:

            st.warning("⚠ Model file not found. Using demo mode.")

            return None

    except Exception as e:

        st.error(f"Error loading model: {e}")

        return None



model = load_model()



MODEL_COLUMNS = [

    "age", "hypertension", "heart_disease", "ever_married", "avg_glucose_level", 

    "bmi", "gender_Male", "work_type_Never_worked", "work_type_Private", 

    "work_type_Self-employed", "work_type_children", "Residence_type_Urban", 

    "smoking_status_formerly smoked", "smoking_status_never smoked", "smoking_status_smokes"

]



# =======================

# 4. Sidebar Navigation

# =======================

st.sidebar.markdown("""

    <div style="text-align: center; margin: 20px 0;">

        <h2 style="color: #38bdf8; margin: 0; font-size: 28px;">🧠 CerebroCare</h2>

        <p style="color: #94a3b8; margin: 5px 0 0 0; font-size: 12px;">Stroke Risk Assessment</p>

    </div>

    <hr style="border-color: #334155; margin: 20px 0;">

""", unsafe_allow_html=True)



page = st.sidebar.radio(

    "Navigation",

    ["🏠 Home", "🔮 Prediction", "⚙ Personalization"],

    label_visibility="collapsed"

)



st.sidebar.markdown("<hr style='border-color: #334155; margin: 30px 0;'>", unsafe_allow_html=True)



# Sidebar info

with st.sidebar:

    st.markdown("""

        <div style="background-color: #1e293b; padding: 16px; border-radius: 8px; border: 1px solid #334155;">

            <p style="font-size: 12px; color: #94a3b8; margin: 0;"><strong>ℹ About</strong></p>

            <p style="font-size: 13px; color: #cbd5e1; margin: 8px 0 0 0;">

                CerebroCare uses advanced machine learning to assess stroke risk based on medical history and lifestyle factors.

            </p>

        </div>

    """, unsafe_allow_html=True)



# =======================

# 5. Page Routing

# =======================

if "🏠" in page:

    from pages import home

    home.render()

elif "🔮" in page:

    from pages import prediction

    prediction.render(model, MODEL_COLUMNS)

elif "⚙" in page:

    from pages import personalization

    personalization.render()
