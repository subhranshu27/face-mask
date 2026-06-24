from huggingface_hub import hf_hub_download
import tensorflow
import streamlit as st
import numpy as np
import cv2
from PIL import Image

@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id="Muthuswamy/Face-mask1",  # 👈 your path
        filename="model.h5"
    )
    return tensorflow.keras.models.load_model(path)

model=load_model()


# Page config
st.set_page_config(page_title="Face Mask Detector", page_icon="😷", layout="centered")

st.title("😷 Face Mask Detection")
st.write("Upload an image to check if the person is wearing a mask.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_array, (128, 128))
    img_scaled = img_resized / 255.0
    img_reshaped = np.reshape(img_scaled, [1, 128, 128, 3])

    # Predict
    with st.spinner("Analyzing..."):
        prediction = model.predict(img_reshaped)
        pred_label = np.argmax(prediction)
        confidence = np.max(prediction) * 100

    # Result
    st.subheader("Result:")
    if pred_label == 1:
        st.success(f"✅ Person IS wearing a mask  ({confidence:.2f}% confidence)")
    else:
        st.error(f"❌ Person is NOT wearing a mask  ({confidence:.2f}% confidence)")

    # Show confidence bar
    st.subheader("Confidence Score:")
    st.progress(int(confidence))
