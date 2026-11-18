# ============================================================
# BRAIN TUMOR AI - GRAD-CAM
# ============================================================

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output",
    "gradcam"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# Classes
# ------------------------------------------------------------

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("Loading model for Grad-CAM...")

MODEL = tf.keras.models.load_model(
    MODEL_PATH
)

print("Grad-CAM model loaded.")


# ------------------------------------------------------------
# Get nested ResNet50
# ------------------------------------------------------------

BASE_MODEL = MODEL.get_layer(
    "resnet50"
)


# ------------------------------------------------------------
# Grad-CAM target layer
# ------------------------------------------------------------

TARGET_LAYER = BASE_MODEL.get_layer(
    "conv5_block3_out"
)


# ------------------------------------------------------------
# Grad-CAM feature model
# ------------------------------------------------------------

RESNET_GRADCAM_MODEL = tf.keras.Model(
    inputs=BASE_MODEL.input,
    outputs=[
        TARGET_LAYER.output,
        BASE_MODEL.output
    ],
    name="resnet_gradcam_model"
)


# ------------------------------------------------------------
# Generate Grad-CAM
# ------------------------------------------------------------

def generate_gradcam(
    image_path,
    output_filename="gradcam_result.jpg"
):
    """
    Generate Grad-CAM for an MRI image.

    Returns:
        predicted_class
        confidence
        output_path
    """

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    original_img = Image.open(
        image_path
    ).convert("RGB")

    display_img = np.array(
        original_img
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    img = original_img.resize(
        (224, 224)
    )

    img_array = np.array(
        img
    ).astype(np.float32)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # --------------------------------------------------------
    # ResNet50 preprocessing
    # --------------------------------------------------------

    img_array = (
        tf.keras.applications.resnet50.preprocess_input(
            img_array
        )
    )

    # --------------------------------------------------------
    # Gradient calculation
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        conv_outputs, resnet_features = (
            RESNET_GRADCAM_MODEL(
                img_array,
                training=False
            )
        )

        # Classifier pipeline

        x = MODEL.get_layer(
            "global_average_pooling"
        )(resnet_features)

        x = MODEL.get_layer(
            "batch_norm"
        )(
            x,
            training=False
        )

        x = MODEL.get_layer(
            "dropout"
        )(
            x,
            training=False
        )

        x = MODEL.get_layer(
            "dense_256"
        )(x)

        x = MODEL.get_layer(
            "batch_norm_2"
        )(
            x,
            training=False
        )

        x = MODEL.get_layer(
            "dropout_2"
        )(
            x,
            training=False
        )

        predictions = MODEL.get_layer(
            "tumor_classifier"
        )(x)

        predicted_index = tf.argmax(
            predictions[0]
        )

        class_score = predictions[
            :,
            predicted_index
        ]

    # --------------------------------------------------------
    # Gradients
    # --------------------------------------------------------

    grads = tape.gradient(
        class_score,
        conv_outputs
    )

    # Global average pooling
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    # Weighted feature maps
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # ReLU
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    heatmap = heatmap / (
        tf.reduce_max(heatmap)
        + tf.keras.backend.epsilon()
    )

    heatmap = heatmap.numpy()

    # --------------------------------------------------------
    # Prediction information
    # --------------------------------------------------------

    predictions_np = (
        predictions.numpy()[0]
    )

    predicted_index = int(
        predicted_index
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = (
        predictions_np[
            predicted_index
        ] * 100
    )

    # --------------------------------------------------------
    # Resize heatmap
    # --------------------------------------------------------

    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        (
            display_img.shape[0],
            display_img.shape[1]
        )
    ).numpy().squeeze()

    # --------------------------------------------------------
    # Save overlay
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        display_img
    )

    plt.imshow(
        heatmap_resized,
        cmap="jet",
        alpha=0.45
    )

    plt.axis("off")

    plt.title(
        f"Grad-CAM\n"
        f"Predicted: {predicted_class} "
        f"({confidence:.2f}%)"
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
        dpi=200
    )

    plt.close()

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "output_path": output_path
    }
    
    # ------------------------------------------------------------
# Local Grad-CAM test
# ------------------------------------------------------------

if __name__ == "__main__":

    test_image = os.path.join(
        BASE_DIR,
        "test_images",
        "test_mri.jpg"
    )

    print("\n" + "=" * 70)
    print("STEP 57 - LOCAL GRAD-CAM TEST")
    print("=" * 70)

    result = generate_gradcam(
        test_image,
        "test_gradcam.jpg"
    )

    print("\nGrad-CAM result:")

    print(
        f"Predicted class : "
        f"{result['predicted_class']}"
    )

    print(
        f"Confidence      : "
        f"{result['confidence']:.2f}%"
    )

    print(
        f"Output saved to : "
        f"{result['output_path']}"
    )

    print("\n" + "=" * 70)
    print("STEP 57 COMPLETED")
    print("=" * 70)