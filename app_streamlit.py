import streamlit as st
import pickle
import numpy as np
import os

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Fraud Detection Indonesia",
    page_icon="🛡️",
    layout="centered"
)

# =========================
# LOAD MODEL (FIX PATH)
# =========================
BASE_DIR = os.path.dirname(__file__)

model_email = pickle.load(open(os.path.join(BASE_DIR, 'model/model_email.pkl'), 'rb'))
model_sms = pickle.load(open(os.path.join(BASE_DIR, 'model/model_sms.pkl'), 'rb'))
model_url = pickle.load(open(os.path.join(BASE_DIR, 'model/model_phishing.pkl'), 'rb'))

vectorizer_email = pickle.load(open(os.path.join(BASE_DIR, 'model/vectorizer_email.pkl'), 'rb'))
vectorizer_sms = pickle.load(open(os.path.join(BASE_DIR, 'model/vectorizer_sms.pkl'), 'rb'))

# =========================
# URL FEATURE
# =========================
def extract_features(url):
    return np.array([[
        len(url),
        url.count('.'),
        int('@' in url),
        int('https' in url)
    ]])

# =========================
# HEADER
# =========================
st.title("🛡️ AI Fraud Detection Indonesia")
st.markdown("""
Deteksi **Email Scam**, **SMS Penipuan**, dan **Phishing URL** secara otomatis menggunakan AI.

💡 Cocok untuk membantu masyarakat menghindari penipuan digital.
""")

st.divider()

# =========================
# PILIH JENIS
# =========================
option = st.selectbox(
    "🔍 Pilih Jenis Deteksi",
    ["📧 Email", "💬 SMS", "🔗 URL"]
)

# =========================
# INPUT USER
# =========================
user_input = st.text_area(
    "✍️ Masukkan teks / URL",
    placeholder="Contoh: Anda mendapatkan hadiah 10 juta..."
)

# =========================
# BUTTON
# =========================
if st.button("🚀 Cek Sekarang"):

    if not user_input.strip():
        st.warning("⚠️ Input tidak boleh kosong!")
    else:

        # EMAIL
        if option == "📧 Email":
            vec = vectorizer_email.transform([user_input])
            pred = model_email.predict(vec)[0]

            if pred == 1:
                st.error("🚨 Email terdeteksi sebagai SPAM / PENIPUAN")
            else:
                st.success("✅ Email Aman")

        # SMS
        elif option == "💬 SMS":
            vec = vectorizer_sms.transform([user_input])
            pred = model_sms.predict(vec)[0]

            if pred == 1:
                st.error("🚨 SMS terdeteksi sebagai PENIPUAN")
            else:
                st.success("✅ SMS Aman")

        # URL
        elif option == "🔗 URL":
            features = extract_features(user_input)
            pred = model_url.predict(features)[0]

            if pred == 1:
                st.error("🚨 URL terdeteksi sebagai PHISHING")
            else:
                st.success("✅ URL Aman")

# =========================
# INFO
# =========================
st.divider()

st.subheader("ℹ️ Tentang Aplikasi")
st.markdown("""
Aplikasi ini menggunakan Machine Learning untuk mendeteksi potensi penipuan digital di Indonesia:

- 📧 Email Scam Detection  
- 💬 SMS Fraud Detection  
- 🔗 Phishing URL Detection  

Dikembangkan untuk membantu pengguna mengenali ancaman digital secara cepat dan mudah.
""")

# =========================
# TIPS
# =========================
st.subheader("💡 Tips Menghindari Penipuan")
st.markdown("""
- Jangan klik link mencurigakan  
- Jangan bagikan OTP atau password  
- Periksa alamat website dengan teliti  
- Waspadai pesan yang terlalu menggiurkan  
""")

# =========================
# FOOTER
# =========================
st.divider()
st.caption("🚀 Project AI Fraud Detection Indonesia | Dibuat dengan Streamlit")
