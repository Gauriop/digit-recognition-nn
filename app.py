"""
app.py  —  Flask backend for Handwritten Digit Recognition
-----------
Folder structure expected:
  your_project/
  ├── app.py
  ├── handwritten.keras     ← exported from Google Colab
  ├── handwritten.pkl       ← exported from Google Colab (optional)
  └── templates/
        └── index.html

IMPORTANT — preprocessing matches Colab Cell 24 exactly:
  1. Grayscale
  2. Resize to 28x28  (no crop, no padding)
  3. Invert:  255 - img_array
  4. tf.keras.utils.normalize(img_array, axis=1)   ← row-wise L2 norm
  5. Reshape to (1, 28, 28)
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

# ── Load model once at startup ────────────────────────────────────────────────
PKL_PATH   = "handwritten (3).pkl"
KERAS_PATH = "handwritten (3).keras"

def find_keras_model():
    """
    Looks for any .keras file in the current directory.
    Handles filenames with spaces like 'handwritten (3).keras'.
    """
    # Try exact name first
    if os.path.exists(KERAS_PATH):
        return KERAS_PATH
    # Scan directory for any .keras file
    for f in os.listdir("."):
        if f.endswith(".keras"):
            print(f"Found keras model: {f}")
            return f
    return None

def load_model():
    keras_path = find_keras_model()
    if keras_path:
        print(f"Loading model from: {keras_path}")
        return tf.keras.models.load_model(keras_path)

    raise FileNotFoundError(
        "No .keras model file found. Place handwritten.keras in the same folder as app.py."
    )

model = load_model()
print("✅ Model ready.")


# ── Preprocessing — matches Colab Cell 24 exactly ─────────────────────────────
def preprocess_image(image_data: str) -> np.ndarray:
    """
    Your Colab Cell 24 does:
        img = Image.open(filename).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img)
        img_array = 255 - img_array                          # invert
        img_array = tf.keras.utils.normalize(img_array, axis=1)  # row-wise L2
        img_input = img_array.reshape(1, 28, 28)

    This function replicates that pipeline exactly from a base64 canvas image.
    """
    # Strip data-URL prefix (e.g. "data:image/png;base64,...")
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    img_bytes = base64.b64decode(image_data)

    # 1. Grayscale
    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    # 2. Resize to 28x28 — exactly like Colab (no padding, no cropping)
    img = img.resize((28, 28), Image.LANCZOS)

    # 3. To numpy float32
    img_array = np.array(img, dtype=np.float32)

    # 4. Invert: canvas = white bg + white/colour ink → need black bg + white ink
    img_array = 255 - img_array

    # 5. Row-wise L2 normalisation — matches Colab exactly
    #    tf.keras.utils.normalize divides each row by its L2 norm (axis=1)
    #    Returns numpy array on some TF versions, tensor on others — np.array() handles both
    normalized = tf.keras.utils.normalize(img_array, axis=1)
    img_array = np.array(normalized, dtype=np.float32)

    # 6. Reshape to (1, 28, 28)
    return img_array.reshape(1, 28, 28).astype(np.float32)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON : { "image": "<base64 PNG>" }
    Returns JSON : {
        "digit":         int,
        "confidence":    float  (0-100),
        "probabilities": [float x 10]
    }
    """
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        arr   = preprocess_image(data["image"])
        probs = model.predict(arr, verbose=0)[0]   # shape (10,)

        digit      = int(np.argmax(probs))
        confidence = float(np.max(probs)) * 100
        probs_list = [round(float(p) * 100, 2) for p in probs]

        return jsonify({
            "digit":         digit,
            "confidence":    round(confidence, 2),
            "probabilities": probs_list,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)