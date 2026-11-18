# ============================================================
# BRAIN TUMOR AI - PREDICTOR
# ============================================================

import os
import numpy as np
import tensorflow as tf
from PIL import Image


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "BrainTumor_ResNet50_GliomaMeningioma.keras"
)


# ------------------------------------------------------------
# Class names
# IMPORTANT: Must match training order
# ------------------------------------------------------------

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ------------------------------------------------------------
# Load model once
# ------------------------------------------------------------

print("Loading Brain Tumor model...")

MODEL = tf.keras.models.load_model(
    MODEL_PATH
)

print("Brain Tumor model loaded successfully.")


# ------------------------------------------------------------
# Prediction function
# ------------------------------------------------------------

def predict_image(image_path):
    """
    Predict brain tumor class from an MRI image.

    Returns:
        predicted_class
        confidence
        probabilities
    """

    # Load image
    image = Image.open(
        image_path
    ).convert("RGB")

    # Resize
    image = image.resize(
        (224, 224)
    )

    # Convert to numpy
    image_array = np.array(
        image
    ).astype(np.float32)

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # ResNet50 preprocessing
    image_array = (
        tf.keras.applications.resnet50.preprocess_input(
            image_array
        )
    )

    # Prediction
    predictions = MODEL.predict(
        image_array,
        verbose=0
    )[0]

    # Predicted class
    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    ) * 100

    # All probabilities
    probabilities = {
        CLASS_NAMES[i]: float(
            predictions[i]
        ) * 100
        for i in range(
            len(CLASS_NAMES)
        )
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities
    }
    
    
    
    
    
    # ------------------------------------------------------------
# Local test
# ------------------------------------------------------------

if __name__ == "__main__":

    test_image = os.path.join(
        BASE_DIR,
        "test_images",
        "test_mri.jpg"
    )

    print("\n" + "=" * 70)
    print("STEP 55 - PREDICTOR TEST")
    print("=" * 70)

    result = predict_image(
        test_image
    )

    print("\nPrediction result:")
    print(
        f"Predicted class : "
        f"{result['predicted_class']}"
    )

    print(
        f"Confidence      : "
        f"{result['confidence']:.2f}%"
    )

    print("\nClass probabilities:")

    for class_name, probability in result[
        "probabilities"
    ].items():

        print(
            f"{class_name:12s}: "
            f"{probability:.2f}%"
        )

    print("\n" + "=" * 70)
    print("STEP 55 COMPLETED")
    print("=" * 70)