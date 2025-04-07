import google.generativeai as genai
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os
import random

# Set your Gemini API key
API_KEY = "AIzaSyAGdptKnrNzFVkMuB5srgpLRplUciFvFoc"
genai.configure(api_key=API_KEY)

# Paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/raspmodel.tflite"
class_indices_path = f"{working_dir}/class_indices.json"
disease_info_path = f"{working_dir}/disease_info.json"

# Load class indices
try:
    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)
except Exception as e:
    st.error(f"Error loading class indices: {e}")
    class_indices = {}

# Load disease info
try:
    with open(disease_info_path, "r") as f:
        disease_info = json.load(f)
except Exception as e:
    st.error(f"Error loading disease information: {e}")
    disease_info = {}

# Load TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"Error loading TFLite model: {e}")
    interpreter = None

# Image Preprocessing
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict Class
def predict_image_class(interpreter, image, class_indices):
    if interpreter is None:
        return "Error: Model not loaded", 0
    img_array = load_and_preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown")
    return predicted_class_name, confidence

# Gemini AI Chatbot
def ai_chatbot(user_query):
    try:
        model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
        response = model.generate_content(user_query)
        return response.text
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Fun facts
fun_facts = [
    "🍅 Tomatoes are technically fruits, not vegetables!",
    "🌱 The largest tomato plant ever recorded grew over 65 feet long!",
    "🚀 NASA has experimented with growing tomatoes in space!",
    "💡 A single tomato plant can produce 200 tomatoes in one season!"
]

# --- Streamlit App ---
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="wide")
st.title('🌿 Plant Disease Classifier & AI Assistant 🚜')

# Tabs
tab1, tab2, tab3 = st.tabs(["📷 Image Classification", "💬 AI Chatbot", "🌾 Crop Recommendation"])

# ---- TAB 1: Image Classification ---- #
with tab1:
    st.header("📷 Upload an Image for Classification")
    uploaded_image = st.file_uploader("Upload a tomato leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((250, 250))
            st.image(resized_img)

        with col2:
            if st.button('🔍 Classify'):
                prediction, confidence = predict_image_class(interpreter, image, class_indices)
                st.success(f'✅ **Prediction:** {prediction} ({confidence:.2f}% Confidence)')

                if prediction in disease_info:
                    symptoms = disease_info[prediction]["symptoms"]
                    remedy = disease_info[prediction]["remedy"]
                    st.markdown(f"**🌿 Symptoms:** {symptoms}")
                    st.markdown(f"**💊 Remedy:** {remedy}")

                st.info(random.choice(fun_facts))

# ---- TAB 2: Chatbot ---- #
with tab2:
    st.header("💬 AI Chatbot")
    st.write("Ask me anything about tomato plant diseases, symptoms, and remedies!")
    user_query = st.text_input("Type your question here...")

    if st.button("Ask AI 🤖"):
        if user_query:
            response = ai_chatbot(user_query)
            st.success(f"🧠 **AI:** {response}")
        else:
            st.warning("⚠️ Please enter a question!")

# ---- TAB 3: Crop Recommendation ---- #
with tab3:
    st.header("🌾 Smart Crop Recommendation")
    st.markdown("Enter your environmental details to get best-fit crop suggestions:")

    temperature = st.slider("🌡️ Temperature (°C)", 10, 45, 25)
    humidity = st.slider("💧 Humidity (%)", 10, 100, 60)
    soil_type = st.selectbox("🌱 Soil Type", ["Sandy", "Loamy", "Clayey", "Black", "Red"])
    season = st.selectbox("🗓️ Season", ["Summer", "Rainy", "Winter"])

    if st.button("🌾 Recommend Crops"):
        # Rule-based logic
        if soil_type == "Loamy" and 20 < temperature < 35 and 40 < humidity < 80:
            recommended = ["Tomato", "Maize", "Chickpea"]
        elif soil_type == "Clayey":
            recommended = ["Rice", "Sugarcane", "Soybean"]
        elif soil_type == "Sandy":
            recommended = ["Groundnut", "Bajra", "Castor"]
        else:
            recommended = ["Wheat", "Cotton", "Barley"]

        st.success("✅ Recommended Crops:")
        for crop in recommended:
            st.markdown(f"- 🌿 **{crop}**")

        st.info("Recommendations are based on basic agricultural guidelines.")

# --- Footer & Tips --- #
with st.expander("🌿 Click for Disease Prevention Tips"):
    st.markdown("""
    - 🚰 Avoid overhead watering to prevent fungal infections.
    - ✂️ Regularly prune to improve air circulation.
    - 🐜 Monitor for pests like whiteflies and mites.
    - 🛑 Rotate crops to avoid soil-borne diseases.
    - 🧼 Disinfect tools after working with infected plants.
    """)

st.markdown("---")
st.markdown("🚀 **Built with Streamlit, TensorFlow Lite & Gemini AI**")
