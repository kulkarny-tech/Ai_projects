import streamlit as st
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import joblib
from PIL import Image

# -----------------------------
# Load trained models and vectorizer
# -----------------------------
try:
    image_model = keras.models.load_model('C:\\Users\\kulka\\my_model.keras')
    text_model = joblib.load("C:\\Users\\kulka\\sgd_model.pkl")
    vectorizer = joblib.load("C:\\Users\\kulka\\vectorizer.pkl")
except Exception as e:
    st.error(f"Error loading models or vectorizer: {e}")

# -----------------------------
# Label mapping for emotions
# -----------------------------
label_map = {
    0: "Love",
    1: "Joy",
    2: "Surprise",
    3: "Sadness",
    4: "Angry",
    5: "Fear"
}

# Reverse mapping for converting text labels to numbers
reverse_label_map = {v.lower(): k for k, v in label_map.items()}

# -----------------------------
# Streamlit UI Setup
# -----------------------------
st.set_page_config(page_title="Multimodal Mood Detection", page_icon="🎭", layout="centered")
st.title("🎭 Multimodal Mood Detection (Image + Text)")
st.markdown("This system detects human emotion by combining **facial expressions** and **text-based sentiment.**")

# ---- IMAGE INPUT ----
st.subheader("🖼 Upload an Image")
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
img_pred_label = None
img_emotion = None

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess image
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    
    # Predict emotion from image
    try:
        img_pred_probs = image_model.predict(image_array)[0]
        img_pred_label = int(np.argmax(img_pred_probs))
        img_emotion = label_map.get(img_pred_label, "Unknown")
        st.success(f"**Image Model Prediction:** {img_emotion}")
    except Exception as e:
        st.error(f"Error processing image: {e}")

# ---- TEXT INPUT ----
st.subheader("💬 Enter Text")
user_text = st.text_area("Type here your sentence or dialogue...")
text_pred_label = None
text_emotion = None

if user_text.strip():
    try:
        # Transform text
        text_vector = vectorizer.transform([user_text])
        
        # Predict using text model
        text_pred_raw = text_model.predict(text_vector)[0]

        # Convert prediction to numeric form (if string)
        if isinstance(text_pred_raw, (np.str_, str)):
            text_emotion = str(text_pred_raw).capitalize()
            text_pred_label = reverse_label_map.get(text_emotion.lower(), None)
        else:
            text_pred_label = int(text_pred_raw)
            text_emotion = label_map.get(text_pred_label, "Unknown")

        st.success(f"**Text Model Prediction:** {text_emotion}")

    except Exception as e:
        st.error(f"Error processing text: {e}")

# ---- FINAL FUSION PREDICTION ----
if img_pred_label is not None and text_pred_label is not None:
    try:
        # Combine both predictions safely using np.unique (no SciPy mode)
        preds = np.array([img_pred_label, text_pred_label])
        values, counts = np.unique(preds, return_counts=True)
        final_pred = values[np.argmax(counts)]

        # Map back to emotion label
        final_emotion = label_map.get(int(final_pred), "Unknown")

        st.subheader(f"🧠 Final Combined Prediction: {final_emotion}")
        st.success(f"Detected Emotion: **{final_emotion}**")

    except Exception as e:
        st.error(f"Error combining predictions: {e}")

# ---- INFO ----
st.markdown("---")
st.caption("Developed by **B. Kulkarny** | Multimodal Mood Detection using Deep Learning")
