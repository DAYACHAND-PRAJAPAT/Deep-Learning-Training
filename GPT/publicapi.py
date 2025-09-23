from transformers import AutoTokenizer, pipeline
from flask import Flask, request, jsonify
from pyngrok import ngrok, conf
import subprocess, time

auth = '335K4N89KMVh2mPE1tAfeewSojv_7qgZy4ruQfNhp1MZD6hnf'

conf.get_default().auth_token = auth

tokenizer = AutoTokenizer.from_pretrained("gpt2")
generator = pipeline("text-generation", model="gpt2")

app = Flask(__name__)

# Tokenization Endpoint
@app.route("/tokenize", methods=['GET', 'POST'])
def tokenize():
    text = request.args.get("text")
    if not text:
        return jsonify({"error": "Please provide input text!!"}), 400

    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)

    return jsonify({
        "Text": text,
        "Tokens": tokens,
        "IDs": ids,
    })


# Text Completion Endpoint
@app.route("/complete", methods=['GET', 'POST'])
def complete():
    text = request.args.get("text")
    if not text:
        return jsonify({"error": "Please provide input text!!"}), 400

    output = generator(text, max_length=30, num_return_sequences=1)

    return jsonify({"GPT": output[0]})


public_url = ngrok.connect(5000)
print("Public URL: ", public_url)

process = subprocess.Popen(
    ["python3", "-m", "flask", "--app", __name__, "run", "--host=0.0.0.0", "--port=5000"]
)

app.run(port=5000)