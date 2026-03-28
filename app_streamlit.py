import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from urllib.parse import urlparse

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Fraud Detection Indonesia",
    page_icon="🛡️",
    layout="centered"
)

# =========================
# CUSTOM CSS
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
# URL FEATURE (FIXED 🔥)
# =========================
feature_names = [
    'url_length', 'dots', 'has_at', 'has_https',
    'hyphen', 'slash', 'digits', 'domain_length', 'path_length'
]

def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path

    return [
        len(url),
        url.count('.'),
        int('@' in url),
        int('https' in url),
        url.count('-'),
        url.count('/'),
        sum(c.isdigit() for c in url),
        len(domain),
        len(path)
    ]

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🛡️ Fraud Detection")
menu = st.sidebar.radio("Pilih Menu", ["🏠 Beranda", "🔍 Deteksi", "ℹ️ Tentang"])

# =========================
# 🏠 BERANDA
# =========================
if menu == "🏠 Beranda":

    st.title("🛡️ AI Fraud Detection Indonesia")
    st.image("assets/cyber security illustration.png", use_container_width=True)

    st.subheader("Lindungi diri dari penipuan digital dengan AI")

    st.markdown("""
Selamat datang di sistem **deteksi penipuan berbasis Artificial Intelligence**.

Aplikasi ini membantu kamu mengenali:
- 📧 Email penipuan  
- 💬 SMS scam  
- 🔗 Link phishing  
""")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("assets/email.png")
        st.markdown("### 📧 Email")
        st.info("Deteksi email penipuan")

    with col2:
        st.image("assets/sms.png")
        st.markdown("### 💬 SMS")
        st.info("Deteksi SMS scam")

    with col3:
        st.image("assets/url.png")
        st.markdown("### 🔗 URL")
        st.info("Deteksi phishing link")

    st.divider()

    st.subheader("📊 Kenapa Ini Penting?")
    st.markdown("""
Penipuan digital semakin meningkat setiap tahun.

Dengan aplikasi ini, kamu bisa:
- menghindari penipuan  
- melindungi data pribadi  
- meningkatkan kesadaran digital  
""")

    st.success("🚀 Lindungi diri kamu mulai sekarang!")

# =========================
# 🔍 DETEKSI
# =========================
elif menu == "🔍 Deteksi":

    st.title("🔍 Deteksi Penipuan")

    option = st.selectbox(
        "Pilih Jenis",
        ["📧 Email", "💬 SMS", "🔗 URL"]
    )

    user_input = st.text_area("Masukkan teks / URL")

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
                features = pd.DataFrame([extract_features(user_input)], columns=feature_names)

                # VALIDASI BIAR AMAN 🔥
                if features.shape[1] != model_url.n_features_in_:
                    st.error("⚠️ Feature tidak sesuai dengan model!")
                    st.stop()

                pred = model_url.predict(features)[0]
                prob = model_url.predict_proba(features)[0]

            confidence = max(prob) * 100

            st.divider()

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
# ℹ️ TENTANG
# =========================
elif menu == "ℹ️ Tentang":

    st.title("ℹ️ Tentang Aplikasi")

    st.markdown("""
Aplikasi ini adalah sistem AI untuk mendeteksi:

- Email Scam  
- SMS Penipuan  
- Phishing URL  
""")

    st.subheader("💡 Tips Keamanan")
    st.markdown("""
- Jangan klik link mencurigakan  
- Jangan bagikan OTP  
- Periksa domain website  
""")

# =========================
# FOOTER
# =========================
st.divider()
st.caption("🚀 Project AI Fraud Detection Indonesia")
