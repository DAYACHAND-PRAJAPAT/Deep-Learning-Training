# streamlit_app.py
import streamlit as st
import requests
from PIL import Image
import io
import os

st.set_page_config(
    page_title="Plant Disease Predictor", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- UI elements and layout ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .main-header {
        color: #2e8b57;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
    }
    .sub-header {
        color: #4682b4;
        text-align: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 1.2em;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-box {
        background-color: #e8f5e9;
        border-left: 6px solid #4CAF50;
        padding: 15px;
        margin-top: 20px;
        border-radius: 5px;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 6px solid #f44336;
        padding: 15px;
        margin-top: 20px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-header">🌿 Plant Disease Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a plant leaf image to detect diseases using a ResNet-18 model.</p>', unsafe_allow_html=True)

# --- Main app logic ---
FLASK_URL = os.environ.get("FLASK_URL", "http://localhost:5000")
PREDICT_ENDPOINT = FLASK_URL.rstrip("/") + "/predict"

uploaded_file = st.file_uploader("Upload an image of a plant leaf (jpg/png).", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Predict"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                with st.spinner("Sending image to Flask API..."):
                    resp = requests.post(PREDICT_ENDPOINT, files=files, timeout=20)
                
                if resp.status_code != 200:
                    st.error(f"API error: {resp.status_code} - {resp.text}")
                else:
                    data = resp.json()
                    st.success(f"Predicted: **{data['predicted_class']}** (index {data['predicted_index']})")
                    st.subheader("Probabilities")
                    probs = data.get("probabilities", {})
                    sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
                    
                    for cls, p in sorted_probs:
                        st.write(f"- {cls}: {p:.3f}")

            except requests.exceptions.ConnectionError:
                st.markdown('<div class="error-box"><p><b>Connection Error:</b> The Flask API is not running or reachable. Please check your network and API server.</p></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to call API: {e}")