import streamlit as st
import pickle
import numpy as np

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Fraud Detection Indonesia",
    page_icon="🛡️",
    layout="centered"
)

# =========================
# LOAD MODEL
# =========================
model_email = pickle.load(open('model_email.pkl', 'rb'))
model_sms = pickle.load(open('model_sms.pkl', 'rb'))
model_url = pickle.load(open('model_phishing.pkl', 'rb'))

vectorizer_email = pickle.load(open('vectorizer_email.pkl', 'rb'))
vectorizer_sms = pickle.load(open('vectorizer_sms.pkl', 'rb'))

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
st.markdown("Deteksi **Email Scam**, **SMS Penipuan**, dan **Phishing URL** secara otomatis menggunakan AI")

st.divider()

# =========================
# MENU PILIHAN
# =========================
option = st.selectbox(
    "🔍 Pilih Jenis Deteksi",
    ["📧 Email", "💬 SMS", "🔗 URL"]
)

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
# INFO SECTION
# =========================
st.divider()

st.subheader("ℹ️ Tentang Aplikasi")
st.markdown("""
Aplikasi ini menggunakan Machine Learning untuk mendeteksi potensi penipuan digital di Indonesia:

- 📧 Email Scam Detection  
- 💬 SMS Fraud Detection  
- 🔗 Phishing URL Detection  

Tujuan: membantu pengguna menghindari penipuan online.
""")

# =========================
# TIPS KEAMANAN
# =========================
st.subheader("💡 Tips Menghindari Penipuan")
st.markdown("""
- Jangan klik link mencurigakan  
- Jangan bagikan OTP atau password  
- Cek alamat website dengan teliti  
- Hindari penawaran yang terlalu bagus untuk dipercaya  
""")

# =========================
# FOOTER
# =========================
st.divider()
st.caption("🚀 Dibuat dengan AI oleh kamu | Project Fraud Detection Indonesia")
