import streamlit as st
import time

# ==========================================
# 1. SETUP SESSION STATE (INGATAN APLIKASI)
# ==========================================
# Kita butuh variabel untuk ingat apakah user sudah klik predict atau belum
if 'prediction_done' not in st.session_state:
    st.session_state['prediction_done'] = False

# Kita butuh variabel untuk tahu kita sedang di halaman mana
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Home Page"

# ==========================================
# 2. FUNGSI UNTUK SETIAP HALAMAN
# ==========================================
def show_home():
    st.title("🏠 Home Page")
    st.write("Selamat datang di aplikasi Stroke Prediction.")

def show_prediction():
    st.title("🔮 Prediction Page")
    st.write("Silakan masukkan data pasien di sini.")
    
    # Simulasi input
    age = st.number_input("Umur", 20, 90)
    
    if st.button("Predict Result"):
        # Simulasi proses loading
        with st.spinner("Sedang memprediksi..."):
            time.sleep(1) # Ceritanya lagi mikir
            
        # --- LOGIKA KUNCI DI SINI ---
        # 1. Tandai bahwa prediksi sudah selesai
        st.session_state['prediction_done'] = True
        
        # 2. Paksa pindah halaman ke 'Personalized Result'
        st.session_state['current_page'] = "Personalized Result"
        
        # 3. Rerun agar aplikasi refresh dan langsung pindah
        st.rerun()

def show_result():
    st.title("📄 Personalized Result")
    st.success("Risiko Stroke: RENDAH")
    st.write("Berdasarkan data Anda, berikut adalah saran kesehatan khusus...")
    st.info("Tab ini hanya muncul karena kamu sudah melakukan prediksi!")
    
    if st.button("Kembali ke Prediksi"):
        st.session_state['current_page'] = "Prediction"
        st.rerun()

# ==========================================
# 3. LOGIKA NAVIGASI (DINAMIS)
# ==========================================

# Tentukan opsi menu apa saja yang boleh muncul
# Awalnya cuma 2 menu
menu_options = ["Home Page", "Prediction"]

# Jika prediksi sudah done, tambahkan menu ke-3
if st.session_state['prediction_done']:
    menu_options.append("Personalized Result")

# --- SIDEBAR / NAVIGATION ---
st.sidebar.header("Navigasi")

# Widget Selectbox harus sinkron dengan session_state['current_page']
# key='current_page' artinya nilai selectbox ini akan otomatis
# mengupdate st.session_state['current_page'] dan sebaliknya.
selection = st.sidebar.selectbox(
    "Pilih Halaman:",
    options=menu_options,
    key='current_page' 
)

# ==========================================
# 4. RENDER HALAMAN SESUAI PILIHAN
# ==========================================
if selection == "Home Page":
    show_home()
elif selection == "Prediction":
    show_prediction()
elif selection == "Personalized Result":
    show_result()
