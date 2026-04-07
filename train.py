"""
train_and_export.py
Run this script once to train the MNIST model and save it as:
  - handwritten.keras   (native Keras format)
  - handwritten.pkl     (pickle wrapper — used by Flask backend)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os

print("=== Loading MNIST dataset ===")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test  = X_test  / 255.0

print("=== Building model ===")
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10,  activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\n=== Training (10 epochs) ===")
model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss    : {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# ── Save as Keras native format ──────────────────────────────────────────────
model.save("handwritten.keras")
print("\n✅  Saved: handwritten.keras")

# ── Save as .pkl (stores the path; Flask loads via tf.keras) ─────────────────
# We pickle a dict so Flask can reconstruct cleanly
pkl_payload = {
    "model_path": os.path.abspath("handwritten.keras"),
    "input_shape": (28, 28),
    "num_classes": 10,
    "labels": list(range(10))
}
with open("handwritten.pkl", "wb") as f:
    pickle.dump(pkl_payload, f)
print("✅  Saved: handwritten.pkl")
print("\nDone! Place both files in your Flask app folder.")