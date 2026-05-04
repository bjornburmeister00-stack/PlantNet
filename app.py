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
    .result-box {
        background-color: #1e3a2f; color: white; padding: 1.8em; border-radius: 12px; margin: 1.2em 0;
    }
    .info-box {
        background-color: #ffffff; padding: 1.6em; border-radius: 10px; border: 2px solid #228B22; margin-top: 1em;
    }
    .footer {text-align: center; color: #555; margin-top: 4em;}
</style>
""", unsafe_allow_html=True)

st.title("🌳 Pflanzen-Detektor – PlantNet-300K")
st.markdown("**Vision Transformer Modell**")

# ====================== MODELL LADEN ======================
@st.cache_resource
def load_model():
    # ←←← HIER DEN GENAUEN MODELLNAMEN EINTRAGEN!
    model_name = "janjibDEV/..."          # Beispiel: "janjibDEV/vit-plantnet300k-base"
    
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return processor, model

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
        st.error("Mapping-Dateien nicht gefunden!")
        return {}, {}

class_to_species, species_to_name = load_mappings()

# ====================== ERKENnung ======================
tab1, tab2 = st.tabs(["🔍 Erkennung starten", "📋 Info"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])
    with col2:
        camera = st.camera_input("Live-Kamera")

    input_image = None
    if uploaded is not None:
        input_image = Image.open(uploaded)
    elif camera is not None:
        input_image = Image.open(camera)

    if input_image is not None:
        st.image(input_image, caption="Dein Bild", use_column_width=True)

        inputs = processor(images=input_image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, class_idx = torch.max(probs, dim=-1)
            
            class_id = str(class_idx.item())
            confidence = confidence.item() * 100

            # Mapping: class_idx → species_id → Name
            species_id = class_to_species.get(class_id)
            predicted_name = species_to_name.get(str(species_id), f"Unbekannte Art (ID: {class_id})")

        st.markdown(f"""
        <div class="result-box">
            <h3>Erkannt: <strong>{predicted_name}</strong></h3>
            <p><strong>Sicherheit:</strong> {confidence:.1f} %</p>
        </div>
        """, unsafe_allow_html=True)

        # Hier kannst du später deine eigenen Tipps + Wiki-Links einbauen

with tab2:
    st.info("Modell basiert auf PlantNet-300K (ca. 1081 Arten).")
    st.write("Für schöne deutsche Namen und Pflegetipps brauchen wir ein zusätzliches Mapping.")

st.markdown("---")
st.markdown('<p class="footer">Schulprojekt 2026 – [Dein Name]</p>', unsafe_allow_html=True)
