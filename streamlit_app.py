import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Plant Disease AI", page_icon="🌿")

st.title("🌿 AI Plant Disease Prediction")
st.write("Upload a tomato leaf image to predict its disease.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_tomato_model.keras")

model = load_model()

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

uploaded_file = st.file_uploader(
    "Choose a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded leaf", width=350)

    resized = image.resize((224, 224))
    image_array = np.array(resized, dtype=np.float32)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]
    top_indices = np.argsort(predictions)[-3:][::-1]

    predicted_index = int(top_indices[0])
    predicted_disease = class_names[predicted_index]
    confidence = float(predictions[predicted_index])

    st.subheader("Prediction Result")
    st.write("**Predicted class:**", predicted_disease)
    st.metric("Model score", f"{confidence:.2%}")

    if confidence < 0.70:
        st.warning(
            "Low model score. Try a clearer image and verify the symptoms."
        )
    else:
        st.info(
            "This is an AI prediction, not a confirmed plant diagnosis."
        )

    st.subheader("Top 3 Predictions")

    labels = [class_names[i] for i in top_indices]
    scores = [float(predictions[i]) * 100 for i in top_indices]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(labels[::-1], scores[::-1])
    ax.set_xlabel("Model score (%)")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
