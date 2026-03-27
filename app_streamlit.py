import streamlit as st
import pickle
import numpy as np
import os

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="🛡️",
    layout="centered"
)

# =========================
# CUSTOM CSS (BIAR KEREN 🔥)
# =========================
st.markdown("""
<style>
.main {
    padding: 2rem;
}
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
.safe {
    background-color: #d4edda;
    color: #155724;
}
.fraud {
    background-color: #f8d7da;
    color: #721c24;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(__file__)

model_email = pickle.load(open(os.path.join(BASE_DIR, 'model/model_email.pkl'), 'rb'))
model_sms = pickle.load(open(os.path.join(BASE_DIR, 'model/model_sms.pkl'), 'rb'))
model_url = pickle.load(open(os.path.join(BASE_DIR, 'model/model_phishing.pkl'), 'rb'))

vectorizer_email = pickle.load(open(os.path.join(BASE_DIR, 'model/vectorizer_email.pkl'), 'rb'))
vectorizer_sms = pickle.load(open(os.path.join(BASE_DIR, 'model/vectorizer_sms.pkl'), 'rb'))

# =========================
# FEATURE URL
# =========================
def extract_features(url):
    return np.array([[
        len(url),
        url.count('.'),
        int('@' in url),
        int('https' in url)
    ]])

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🛡️ Fraud Detection")
menu = st.sidebar.radio("Pilih Menu", ["Deteksi", "Tentang"])

# =========================
# HALAMAN UTAMA
# =========================
if menu == "Deteksi":

    st.title("🛡️ AI Fraud Detection Indonesia")
    st.caption("Deteksi Email, SMS, dan URL Penipuan dengan AI")

    st.divider()

    option = st.selectbox(
        "🔍 Pilih Jenis Deteksi",
        ["📧 Email", "💬 SMS", "🔗 URL"]
    )

    user_input = st.text_area(
        "✍️ Masukkan teks / URL",
        placeholder="Contoh: Selamat! Anda mendapatkan hadiah..."
    )

    if st.button("🚀 Cek Sekarang"):

        if not user_input.strip():
            st.warning("⚠️ Input tidak boleh kosong!")
        else:

            if option == "📧 Email":
                vec = vectorizer_email.transform([user_input])
                pred = model_email.predict(vec)[0]
                prob = model_email.predict_proba(vec)[0]

            elif option == "💬 SMS":
                vec = vectorizer_sms.transform([user_input])
                pred = model_sms.predict(vec)[0]
                prob = model_sms.predict_proba(vec)[0]

            elif option == "🔗 URL":
                features = extract_features(user_input)
                pred = model_url.predict(features)[0]
                prob = model_url.predict_proba(features)[0]

            confidence = max(prob) * 100

            st.divider()

            # =========================
            # RESULT CARD
            # =========================
            if pred == 1:
                st.markdown(f"""
                <div class="result-box fraud">
                    🚨 TERDETEKSI PENIPUAN <br>
                    Confidence: {confidence:.2f}%
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box safe">
                    ✅ AMAN <br>
                    Confidence: {confidence:.2f}%
                </div>
                """, unsafe_allow_html=True)

# =========================
# HALAMAN TENTANG
# =========================
elif menu == "Tentang":

    st.title("ℹ️ Tentang Aplikasi")

    st.markdown("""
Aplikasi ini adalah sistem **AI Fraud Detection** yang mampu mendeteksi:

- 📧 Email Scam  
- 💬 SMS Penipuan  
- 🔗 Phishing URL  

Dibuat menggunakan Machine Learning untuk membantu masyarakat menghindari penipuan digital di Indonesia.
""")

    st.subheader("💡 Tips Keamanan")
    st.markdown("""
- Jangan klik link mencurigakan  
- Jangan bagikan OTP atau password  
- Periksa domain website dengan teliti  
- Hindari tawaran yang terlalu menggiurkan  
""")

# =========================
# FOOTER
# =========================
st.divider()
st.caption("🚀 Project AI Fraud Detection Indonesia")
