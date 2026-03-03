"""
MediAlert Model API
===================
Standalone REST API for pneumonia detection from chest X-ray images.
Uses a trained HOG + SVM classifier (82.1% test accuracy).


Endpoints:
    GET  /health       - Health check
    POST /predict      - Predict NORMAL or PNEUMONIA from an uploaded image
"""

import os
import io
import time
import joblib
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image
from skimage.feature import hog

app = Flask(__name__)

# ── Model loading ─────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "classical_svm.pkl"
_model_bundle = None


def load_model():
    """Load the SVM model bundle (lazy, cached after first call)."""
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


# ── Feature extraction ────────────────────────────────────────────────────────

IMAGE_SIZE = (64, 64)


def extract_features(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes to a HOG feature vector.

    Args:
        image_bytes: Raw image bytes from the uploaded file.

    Returns:
        1-D numpy array of HOG features.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")   # Grayscale
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0

    features = hog(
        arr,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return features


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    model_ok = MODEL_PATH.exists()
    return jsonify({
        "status": "healthy" if model_ok else "model_missing",
        "model": "HOG+SVM",
        "model_file": str(MODEL_PATH.name),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict pneumonia from a chest X-ray image.

    Expects multipart/form-data with field name 'image'.

    Returns JSON:
        {
            "prediction": "PNEUMONIA" | "NORMAL",
            "confidence": float (0-1),
            "probabilities": {"NORMAL": float, "PNEUMONIA": float},
            "processing_time_ms": int,
            "model_version": str
        }
    """
    start = time.time()

    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use field name 'image'."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Validate extension
    allowed = {"jpg", "jpeg", "png"}
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type: {ext}. Use JPG or PNG."}), 400

    try:
        image_bytes = file.read()
        features = extract_features(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 422

    try:
        bundle = load_model()
        svm = bundle["model"]
        scaler = bundle["scaler"]
        classes = bundle["classes"]          # ["NORMAL", "PNEUMONIA"]
    except Exception as e:
        return jsonify({"error": f"Model load failed: {str(e)}"}), 500

    # Scale + predict
    feat_scaled = scaler.transform([features])
    pred_idx = int(svm.predict(feat_scaled)[0])
    prediction = pred_idx if isinstance(pred_idx, str) else classes[pred_idx] if isinstance(pred_idx, int) else pred_idx

    # Probabilities
    if hasattr(svm, "predict_proba"):
        proba = svm.predict_proba(feat_scaled)[0]
        probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        confidence = round(float(max(proba)), 4)
    else:
        probabilities = {"NORMAL": 0.0, "PNEUMONIA": 0.0}
        probabilities[prediction] = 1.0
        confidence = 1.0

    elapsed_ms = int((time.time() - start) * 1000)

    return jsonify({
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "processing_time_ms": elapsed_ms,
        "model_version": "SVM-HOG-v1",
        "heatmap": None,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
