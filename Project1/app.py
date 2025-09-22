from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return "Movie Review Sentiment API is running! Use POST /predict with JSON {\"text\": \"your review\"}"

@app.route("/predict", methods=["GET","POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    X = vectorizer.transform([text])
    pred = model.predict(X)[0]

    return jsonify({"Sentiment" : pred})

if __name__ == "__main__":
    app.run(debug=True)