# flask_app.py
from flask import Flask, request, jsonify
import torch
import io
from utils import preprocess_image_bytes, PLANT_DISEASE_CLASSES
from model import PlantDiseaseResNet18
import os

app = Flask(__name__)

# Config
MODEL_PATH = os.environ.get("MODEL_PATH", "saved_models/plant_disease_model_small.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
def load_model(path=MODEL_PATH, device=DEVICE):
    num_classes = len(PLANT_DISEASE_CLASSES)
    model = PlantDiseaseResNet18(num_classes=num_classes)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt)
    model.to(device).eval()
    return model

if os.path.exists(MODEL_PATH):
    model = load_model()
else:
    model = None
    print(f"Warning: Model file not found at {MODEL_PATH}. Prediction endpoint will be disabled.")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model and ensure the path is correct."}), 503
    
    if 'file' not in request.files:
        return jsonify({"error":"no file uploaded"}), 400
    file = request.files['file']
    img_bytes = file.read()
    try:
        tensor = preprocess_image_bytes(img_bytes)
    except Exception as e:
        return jsonify({"error":"invalid image", "detail": str(e)}), 400
    tensor = tensor.to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().tolist()[0]
        pred_idx = int(torch.argmax(outputs, dim=1).cpu().item())
    return jsonify({
        "predicted_class": PLANT_DISEASE_CLASSES[pred_idx],
        "predicted_index": pred_idx,
        "probabilities": {PLANT_DISEASE_CLASSES[i]: float(probs[i]) for i in range(len(probs))}
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)