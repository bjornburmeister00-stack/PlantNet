import streamlit as st
from PIL import Image
import torch
import json
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ====================== DESIGN ======================
st.set_page_config(page_title="Pflanzen-Detektor", page_icon="🌳", layout="wide")

st.markdown("""
<style>
    .main {background-color: #f0f8f0;}
    h1 {color: #228B22; text-align: center;}
    .result-box {background-color: #1e3a2f; color: white; padding: 1.8em; border-radius: 12px; margin: 1.2em 0;}
    .footer {text-align: center; color: #555; margin-top: 4em; font-size: 0.95em;}
</style>
""", unsafe_allow_html=True)

st.title("🌳 Pflanzen-Detektor – PlantNet-300K")
st.markdown("**ViT-Modell (Hugging Face)**")

# ====================== MODELL LADEN ======================
@st.cache_resource
def load_model():
    try:
        model_name = "janjibDEV/vit-plantnet300k"
        
        st.info("Lade Modell von Hugging Face...")
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        st.success("✅ Modell erfolgreich geladen!")
        return processor, model
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        st.info("Tipp: Stelle sicher, dass 'torch' und 'torchvision' in requirements.txt sind.")
        return None, None

processor, model = load_model()

# ====================== MAPPING LADEN ======================
@st.cache_data
def load_mappings():
    try:
        with open("class_idx_to_species_id.json", "r", encoding="utf-8") as f:
            class_to_species = json.load(f)
        with open("plantnet300K_species_id_2_name.json", "r", encoding="utf-8") as f:
            species_to_name = json.load(f)
        return class_to_species, species_to_name
    except:
        return {}, {}

class_to_species, species_to_name = load_mappings()

# ====================== ERKENnung ======================
tab1, tab2 = st.tabs(["🔍 Erkennung starten", "📋 Info"])

with tab1:
    st.subheader("Foto hochladen oder Kamera nutzen")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])
    with col2:
        camera_file = st.camera_input("Live-Kamera")

    input_image = None
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
    elif camera_file is not None:
        input_image = Image.open(camera_file)

    if input_image is not None and processor is not None and model is not None:
        st.image(input_image, caption="Dein Bild", use_column_width=True)

        inputs = processor(images=input_image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, class_idx = torch.max(probs, dim=-1)
            
            class_id = str(class_idx.item())
            confidence = confidence.item() * 100

            species_id = class_to_species.get(class_id)
            predicted_name = species_to_name.get(str(species_id), f"Unbekannte Art (ID: {class_id})")

        st.markdown(f"""
        <div class="result-box">
            <h3>Erkannt: <strong>{predicted_name}</strong></h3>
            <p><strong>Sicherheit:</strong> {confidence:.1f} %</p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.info("Modell: janjibDEV/vit-plantnet300k (PlantNet-300K)")

st.markdown("---")
st.markdown('<p class="footer">Schulprojekt 2026 – [Dein Name]</p>', unsafe_allow_html=True)
