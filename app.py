import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fire Detection System",
    page_icon="🔥",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>

    .main {
        background-color: #0E1117;
    }

    .title {
        text-align: center;
        font-size: 45px;
        font-weight: bold;
        color: #FF4B4B;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        color: white;
        margin-bottom: 30px;
    }

    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-size: 20px;
        border-radius: 10px;
        height: 3em;
        border: none;
    }

    .result-fire {
        color: red;
        font-size: 35px;
        font-weight: bold;
        text-align: center;
    }

    .result-safe {
        color: green;
        font-size: 35px;
        font-weight: bold;
        text-align: center;
    }

    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown(
    '<p class="title">🔥 Fire Detection System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Upload an image and click Predict</p>',
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(
    "models/fire_detection_model.h5"
)

IMG_SIZE = 128

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- IMAGE DISPLAY ----------------
if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    # ---------------- PREDICT BUTTON ----------------
    if st.button("Predict"):

        # Resize image
        resized_img = img.resize((IMG_SIZE, IMG_SIZE))

        # Convert image to array
        img_array = image.img_to_array(resized_img)

        # Normalize image
        img_array = img_array / 255.0

        # Expand dimensions
        img_array = np.expand_dims(img_array, axis=0)

        # ---------------- PREDICTION ----------------
        prediction = model.predict(img_array)

        confidence = prediction[0][0]

        st.write("")

        # -------------------------------------------------
        # IMPORTANT:
        # fire = 0
        # no_fire = 1
        # -------------------------------------------------

        if confidence > 0.5:

            st.markdown(
                '<p class="result-safe">✅ NO FIRE DETECTED</p>',
                unsafe_allow_html=True
            )

            st.success(
                f"Confidence: {confidence*100:.2f}%"
            )

        else:

            st.markdown(
                '<p class="result-fire">🔥 FIRE DETECTED</p>',
                unsafe_allow_html=True
            )

            st.error(
                f"Confidence: {(1-confidence)*100:.2f}%"
            )