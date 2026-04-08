""" app.py — Flask backend for Handwritten Digit & Letter Recognition
-----------
Supports both digit recognition (MNIST) and letter recognition (custom EMNIST model)
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle
import os
from PIL import Image, ImageOps
import io
import base64

app = Flask(__name__)
CORS(app)

# ============= DIGIT RECOGNITION (MNIST) =============

def find_keras_model():
    """Find the saved Keras digit model file."""
    candidates = [
        os.path.expanduser("~/digit-recognition-nn/handwritten.keras"),
        os.path.expanduser("~/digit-recognition-nn/handwritten.h5"),
        "./handwritten.keras",
        "./handwritten.h5",
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"Found digit model: {path}")
            return path
    return None

def load_digit_labels():
    """Load class labels for digit recognition."""
    label_candidates = [
        os.path.expanduser("~/digit-recognition-nn/handwritten.pkl"),
        "./handwritten.pkl",
    ]
    for path in label_candidates:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return list(range(10))  # Default: 0-9

# Load digit model
digit_model = tf.keras.models.load_model(find_keras_model(), compile=False)
digit_labels = load_digit_labels()
print("✅ Digit model ready. Input shape:", digit_model.input_shape)

# ============= LETTER RECOGNITION (Custom EMNIST Model) =============

def find_letter_model():
    """Find the saved Keras letter model file."""
    candidates = [
        os.path.expanduser("~/digit-recognition-nn/handwriting_letters.keras"),
        "./handwriting_letters.keras",
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"Found letter model: {path}")
            return path
    return None

def load_letter_labels():
    """Load class labels for letter recognition."""
    label_candidates = [
        os.path.expanduser("~/digit-recognition-nn/handwriting_labels.pkl"),
        "./handwriting_labels.pkl",
    ]
    for path in label_candidates:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return {i: chr(65 + i) for i in range(26)}  # Default: 0->A ... 25->Z

# Load letter model
letter_model = tf.keras.models.load_model(find_letter_model(), compile=False)
letter_labels = load_letter_labels()
print("✅ Letter model ready. Input shape:", letter_model.input_shape)

# ============= ROUTES =============

@app.route("/")
def index():
    """Digit recognition page."""
    return render_template("index.html")

@app.route("/handwriting")
def handwriting():
    """Letter recognition page."""
    return render_template("handwriting.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Predict digit from canvas image."""
    try:
        data = request.get_json()
        img_data = base64.b64decode(data["image"].split(",")[1])
        image = Image.open(io.BytesIO(img_data)).convert("L")  # Grayscale
        image = np.array(image) / 255.0
        image = image.reshape(1, 28, 28, 1)

        prediction = digit_model.predict(image, verbose=0)
        predicted_label = digit_labels[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

        return jsonify({
            "prediction": str(predicted_label),
            "confidence": round(confidence, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict_handwriting", methods=["POST"])
def predict_handwriting():
    """Predict handwritten letter using custom EMNIST model."""
    try:
        data = request.get_json()
        img_data = base64.b64decode(data["image"].split(",")[1])

        # Convert to grayscale
        image = Image.open(io.BytesIO(img_data)).convert("L")

        # Resize to 28x28 (what the model expects)
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # Normalize to 0-1
        img_array = np.array(image, dtype="float32") / 255.0

        # Reshape for model input: (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        prediction = letter_model.predict(img_array, verbose=0)
        predicted_index = int(np.argmax(prediction))
        predicted_letter = letter_labels[predicted_index]
        confidence = float(np.max(prediction)) * 100

        return jsonify({
            "text": predicted_letter,
            "confidence": round(confidence, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)