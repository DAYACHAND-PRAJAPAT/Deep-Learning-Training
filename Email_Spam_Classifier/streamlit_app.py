import streamlit as st
import requests
import os

st.set_page_config(
    page_title="Spam Classifier", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .main-header {
        color: #e55959;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .sub-header {
        color: #7d96a7;
        text-align: center;
        font-size: 1.2em;
        margin-top: -10px;
    }
    .stButton>button {
        background-color: #e55959;
        color: white;
        font-size: 1.2em;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        transition-duration: 0.4s;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #d32f2f;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .result-box {
        background-color: #e8f5e9;
        border-left: 6px solid #4CAF50;
        padding: 15px;
        margin-top: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .error-box {
        background-color: #ffebee;
        border-left: 6px solid #f44336;
        padding: 15px;
        margin-top: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stTextArea label {
        font-weight: bold;
        font-size: 1.1em;
    }
    .stTextArea textarea {
        border-radius: 10px;
        padding: 10px;
        border: 2px solid #ccc;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<p class="main-header">📧 Spam Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enter a message to instantly see if it is spam or not.</p>', unsafe_allow_html=True)

FLASK_URL = os.environ.get("FLASK_URL", "https://rozanne-unpiloted-entrappingly.ngrok-free.dev")
PREDICT_ENDPOINT = FLASK_URL.rstrip("/") + "/predict"

text_input = st.text_area("Your message here:")

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("Classify"):
        if not text_input:
            st.warning("Please enter a message to classify.")
        else:
            try:
                with st.spinner("Classifying..."):
                    resp = requests.post(PREDICT_ENDPOINT, json={"text": text_input}, timeout=20)
                
                if resp.status_code != 200:
                    st.error(f"API error: {resp.status_code} - {resp.text}")
                else:
                    data = resp.json()
                    prediction = data.get("Prediction")

                    if prediction == "spam":
                        st.markdown(f'<div class="error-box"><p><b>Prediction:</b> 🚫 Spam</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-box"><p><b>Prediction:</b> ✅ Ham (Not Spam)</p></div>', unsafe_allow_html=True)

            except requests.exceptions.ConnectionError:
                st.markdown('<div class="error-box"><p><b>Connection Error:</b> The Flask API is not running or reachable. Please check your network and API server.</p></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to call API: {e}")