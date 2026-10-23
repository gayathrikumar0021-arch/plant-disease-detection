
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# App configuration
st.set_page_config(
    page_title="Tomato Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Tomato Leaf Disease Detection")
st.write("Upload a tomato leaf image to predict its disease.")

# Load the trained model
MODEL_PATH = Path(__file__).parent / "best_tomato_model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

# Class names in the same order as the training dataset
class_names = [
    "Bacterial Spot",
    "Early Blight",
    "Healthy",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites Two-spotted Spider Mite",
    "Target Spot",
    "Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus"
]

# Upload an image
uploaded_file = st.file_uploader(
    "Choose a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Leaf", width=350)

    if st.button("Detect Disease", type="primary"):
        with st.spinner("Analyzing leaf image..."):
            # Resize image to match model input
            image = image.resize((224, 224))
            image_array = np.array(image, dtype=np.float32)

            # Apply the same preprocessing used during training
            image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                image_array
            )
            image_array = np.expand_dims(image_array, axis=0)

            # Predict the disease
            predictions = model.predict(image_array, verbose=0)[0]
            predicted_index = int(np.argmax(predictions))
            confidence = float(predictions[predicted_index])

        st.subheader("Prediction Result")
        st.success(f"Predicted Class: {class_names[predicted_index]}")
        st.metric("Model Confidence", f"{confidence:.2%}")

        # Display the top 3 predictions
        st.subheader("Top 3 Predictions")
        top_indices = np.argsort(predictions)[-3:][::-1]

        for index in top_indices:
            st.write(
                f"**{class_names[index]}** — "
                f"{predictions[index]:.2%}"
            )
            st.progress(float(predictions[index]))

        st.caption(
            "AI predictions are estimates. Confirm uncertain results "
            "with a plant disease expert."
        )
