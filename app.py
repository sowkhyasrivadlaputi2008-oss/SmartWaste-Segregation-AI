import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Smart Waste Segregation AI",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Smart Waste Segregation AI")
st.write(
    "Upload a waste image to view the model's prediction "
    "and learn how to dispose of the item responsibly."
)

# ---------------- MODEL LOADING ----------------
MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "waste-classification-model.h5"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

try:
    model = load_model()
    st.success("AI model loaded successfully!")
except Exception as error:
    st.error("Unable to load the AI model.")
    st.code(str(error))
    st.stop()

# IMPORTANT:
# These labels are provisional.
# Verify their order against the model's original training code.
CLASS_NAMES = [
    "Cardboard",
    "Glass",
    "Metal",
    "Paper",
    "Plastic",
    "Trash"
]

DISPOSAL_TIPS = {
    "Cardboard": (
        "Keep cardboard dry and clean. "
        "Flatten boxes and place them in paper recycling."
    ),
    "Glass": (
        "Handle glass carefully. Separate it from other waste "
        "and follow your local glass recycling rules."
    ),
    "Metal": (
        "Empty metal containers and rinse them when practical "
        "before placing them in metal recycling."
    ),
    "Paper": (
        "Keep paper clean and dry. Recycle it where accepted."
    ),
    "Plastic": (
        "Empty and clean the plastic item when practical. "
        "Check whether your local recycling service accepts it."
    ),
    "Trash": (
        "Place non-recyclable waste in the appropriate "
        "general-waste bin."
    )
}

# ---------------- IMAGE UPLOAD ----------------
st.subheader("Upload Waste Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded waste image",
            use_container_width=True
        )

        # Read the model's expected input dimensions.
        input_shape = model.input_shape

        if isinstance(input_shape, list):
            st.error("This app expects a single-input model.")
            st.stop()

        height = input_shape[1]
        width = input_shape[2]

        if height is None or width is None:
            st.error("The model's input image size is unknown.")
            st.stop()

        # Resize to the model's expected dimensions.
        resized_image = image.resize(
            (int(width), int(height))
        )

        image_array = np.asarray(
            resized_image,
            dtype=np.float32
        )

        # This normalization must match the original training.
        # Confirm it before relying on prediction accuracy.
        image_array = image_array / 255.0

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # ---------------- PREDICTION ----------------
        with st.spinner("Analyzing image..."):
            predictions = model.predict(
                image_array,
                verbose=0
            )

        scores = np.asarray(predictions)[0]

        if len(scores) != len(CLASS_NAMES):
            st.error(
                "The number of model outputs does not match "
                "the number of configured class labels."
            )
            st.stop()

        if not np.all(np.isfinite(scores)):
            st.error("The model returned invalid prediction scores.")
            st.stop()

        predicted_index = int(np.argmax(scores))
        predicted_category = CLASS_NAMES[predicted_index]
        confidence = float(scores[predicted_index])

        st.subheader("Model Output Scores")

        for name, score in zip(CLASS_NAMES, scores):
            st.write(f"**{name}:** {float(score):.4f}")

        st.subheader("Predicted Category")
        st.write(f"### {predicted_category}")

        st.metric(
            "Model confidence score",
            f"{confidence * 100:.2f}%"
        )

        st.info(
            "These results depend on the model's original class "
            "order and preprocessing. A high score does not "
            "guarantee a correct classification."
        )

        st.subheader("♻️ Disposal Recommendation")
        st.write(DISPOSAL_TIPS[predicted_category])

    except Exception as error:
        st.error("Could not process this image.")
        st.code(str(error))

# ---------------- ABOUT PROJECT ----------------
with st.expander("About the Project"):
    st.write("""
    Smart Waste Segregation AI uses a trained TensorFlow
    image-classification model to predict a waste category.

    Categories:
    - Cardboard
    - Glass
    - Metal
    - Paper
    - Plastic
    - Trash

    The predicted category is used to display a disposal tip.
    Prediction accuracy depends on the training data,
    correct class mapping, and image preprocessing.
    """)

# ---------------- PROJECT EVALUATION ----------------
with st.expander("Project Evaluation"):
    st.write("**Technology used:** Python, TensorFlow, Streamlit, NumPy and Pillow.")
    st.write("**Input:** An uploaded waste image.")
    st.write("**Output:** Predicted category, model scores and disposal recommendation.")
    st.write("**Limitation:** The original model's class mapping and preprocessing must be verified.")