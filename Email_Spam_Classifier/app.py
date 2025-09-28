from flask import Flask, request, jsonify
import joblib
import os
import nltk
from nltk.corpus import stopwords
import string
from pyngrok import ngrok, conf

NGROK_AUTH_TOKEN = '335K4N89KMVh2mPE1tAfeewSojv_7qgZy4ruQfNhp1MZD6hnf'
conf.get_default().auth_token = NGROK_AUTH_TOKEN

app = Flask(__name__)

MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
    print("Model or vectorizer file not found. Please run main.py first to train the model.")
    exit()

model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

def preprocess_text(text):
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.lower().split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    processed_text = preprocess_text(text)
    X = vectorizer.transform([processed_text])
    pred = model.predict(X)[0]
    
    return jsonify({"Prediction": pred})

if __name__ == "__main__":
    try:
        public_url = ngrok.connect(5000)
        print("ngrok public URL:", public_url)
    except Exception as e:
        print(f"Error creating ngrok tunnel: {e}")
        public_url = None
    
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)