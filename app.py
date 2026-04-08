"""
app.py  —  Flask backend for Handwritten Digit Recognition
-----------
Folder structure expected:
  your_project/
  ├── app.py
  ├── handwritten (3).keras
  ├── handwritten (3).pkl
  └── templates/
        └── index.html

KEY FIX: Canvas sends black-bg + white-ink images.
  DO NOT invert. Instead, crop → center → pad → normalize.
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import pickle
import base64
import io
import os
from PIL import Image

app = Flask(__name__)
CORS(app)

# ── Load model & labels ───────────────────────────────────────────────────────
KERAS_PATH = "handwritten (3).keras"
PKL_PATH   = "handwritten (3).pkl"

def find_keras_model():
    if os.path.exists(KERAS_PATH):
        return KERAS_PATH
    for f in os.listdir("."):
        if f.endswith(".keras"):
            print(f"Found keras model: {f}")
            return f
    return None

def load_labels():
    for p in [PKL_PATH, "handwritten (3).pkl"]:
        if os.path.exists(p):
            with open(p, "rb") as f:
                labels = pickle.load(f)
            print(f"✅ Labels loaded from {p}: {labels}")
            return labels
    # Default fallback
    return [str(i) for i in range(10)]

model  = tf.keras.models.load_model(find_keras_model())
labels = load_labels()
print("✅ Model ready. Input shape:", model.input_shape)


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(image_data: str) -> np.ndarray:
    """
    Canvas = black background (#0c0c14) + white ink.
    MNIST   = black background + white digit, 28x28, centered.

    Steps:
      1. Decode base64 → grayscale PIL image
      2. Resize canvas to 28x28
      3. NO inversion (canvas already has white ink on black)
      4. Crop bounding box of drawn content
      5. Resize crop to 20x20, pad to 28x28 (centers digit like MNIST)
      6. Row-wise L2 normalize (axis=1) — matches Colab training
      7. Reshape to (1, 28, 28)
    """
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    # Resize to 28x28 first for bounding-box detection
    img = img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)

    # ── Center the digit (MNIST-style) ────────────────────────────────────────
    threshold = 30  # ignore near-black background noise
    mask = img_array > threshold

    if mask.any():
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        cropped = img_array[rmin:rmax+1, cmin:cmax+1]

        # Resize crop to 20x20, leaving a 4-px border all around
        pil_crop    = Image.fromarray(cropped.astype(np.uint8))
        pil_resized = pil_crop.resize((20, 20), Image.LANCZOS)

        # Place in center of 28x28 black canvas
        img_array = np.zeros((28, 28), dtype=np.float32)
        img_array[4:24, 4:24] = np.array(pil_resized, dtype=np.float32)

    # ── Row-wise L2 normalisation (matches Colab Cell 24) ────────────────────
    normalized = tf.keras.utils.normalize(img_array, axis=1)
    img_array  = np.array(normalized, dtype=np.float32)

    return img_array.reshape(1, 28, 28).astype(np.float32)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        arr   = preprocess_image(data["image"])
        probs = model.predict(arr, verbose=0)[0]   # shape (10,)

        digit      = int(np.argmax(probs))
        confidence = float(np.max(probs)) * 100
        probs_list = [round(float(p) * 100, 2) for p in probs]

        # Use pkl labels if available
        label = labels[digit] if labels else str(digit)

        return jsonify({
            "digit":         digit,
            "label":         label,
            "confidence":    round(confidence, 2),
            "probabilities": probs_list,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None, "labels": labels})


if __name__ == "__main__":
    app.run(debug=True, port=5000)