import tensorflow as tf
import numpy as np
import os
from PIL import Image

print("=" * 70)
print("STEP 53 - LOCAL MRI PREDICTION TEST")
print("=" * 70)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "BrainTumor_ResNet50_GliomaMeningioma.keras"
)

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "test_images",
    "test_mri.jpg"
)

# ------------------------------------------------------------
# Classes
# ------------------------------------------------------------

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# ------------------------------------------------------------
# Check files
# ------------------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Test image not found: {IMAGE_PATH}"
    )

print("\nModel:")
print(MODEL_PATH)

print("\nImage:")
print(IMAGE_PATH)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")

# ------------------------------------------------------------
# Load image
# ------------------------------------------------------------

print("\nLoading MRI...")

image = Image.open(
    IMAGE_PATH
).convert("RGB")

print(
    f"Original image size: {image.size}"
)

# ------------------------------------------------------------
# Resize
# ------------------------------------------------------------

image = image.resize(
    (224, 224)
)

# ------------------------------------------------------------
# Convert to numpy
# ------------------------------------------------------------

image_array = np.array(
    image
).astype(np.float32)

# Add batch dimension

image_array = np.expand_dims(
    image_array,
    axis=0
)

# ------------------------------------------------------------
# ResNet50 preprocessing
# ------------------------------------------------------------

image_array = (
    tf.keras.applications.resnet50.preprocess_input(
        image_array
    )
)

print(
    f"Input shape: {image_array.shape}"
)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

print("\nRunning prediction...")

predictions = model.predict(
    image_array,
    verbose=0
)[0]

predicted_index = np.argmax(
    predictions
)

predicted_class = class_names[
    predicted_index
]

confidence = (
    predictions[predicted_index] * 100
)

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

print(
    f"\nPredicted class : {predicted_class}"
)

print(
    f"Confidence      : {confidence:.2f}%"
)

print("\nClass probabilities:")

for i, class_name in enumerate(class_names):

    print(
        f"{class_name:12s}: "
        f"{predictions[i] * 100:.2f}%"
    )

print("\n" + "=" * 70)
print("STEP 53 COMPLETED")
print("=" * 70)