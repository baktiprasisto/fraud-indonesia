import streamlit as st
import pickle
import numpy as np

# Load models
model_email = pickle.load(open('model/model_email.pkl', 'rb'))
model_sms = pickle.load(open('model/model_sms.pkl', 'rb'))
model_url = pickle.load(open('model/model_phishing.pkl', 'rb'))

vectorizer_email = pickle.load(open('model/vectorizer_email.pkl', 'rb'))
vectorizer_sms = pickle.load(open('model/vectorizer_sms.pkl', 'rb'))

# URL feature
def extract_features(url):
    return np.array([[
        len(url),
        url.count('.'),
        int('@' in url),
        int('https' in url)
    ]])

# UI
st.title("🔥 AI Fraud Detection Indonesia")

option = st.selectbox("Pilih Jenis Deteksi", ["Email", "SMS", "URL"])

user_input = st.text_area("Masukkan teks / URL")

if st.button("Cek Sekarang"):

    if option == "Email":
        vec = vectorizer_email.transform([user_input])
        pred = model_email.predict(vec)[0]

        if pred == 1:
            st.error("🚨 Spam Email")
        else:
            st.success("✅ Email Aman")

    elif option == "SMS":
        vec = vectorizer_sms.transform([user_input])
        pred = model_sms.predict(vec)[0]

        if pred == 1:
            st.error("🚨 SMS Spam")
        else:
            st.success("✅ SMS Aman")

    elif option == "URL":
        features = extract_features(user_input)
        pred = model_url.predict(features)[0]

        if pred == 1:
            st.error("🚨 Phishing URL")
        else:
            st.success("✅ URL Aman")
