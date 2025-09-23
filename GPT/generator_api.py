from transformers import AutoTokenizer, pipeline
from flask import Flask, request, jsonify

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
