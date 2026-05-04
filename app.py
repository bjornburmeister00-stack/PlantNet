import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

st.set_page_config(page_title="Pflanzen-Detektor", page_icon="🌳", layout="wide")

st.title("🌳 Pflanzen-Detektor – PlantNet-300K")
st.markdown("**Letzter Versuch**")

@st.cache_resource
def load_model():
    try:
        model_name = "janjibDEV/vit-plantnet300k"
        
        # Sehr aggressiver Ladeversuch
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True,
            trust_remote_code=True
        )
        
        # Letzter Fix-Versuch
        if hasattr(model.config, "id2label"):
            model.config.id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
        
        st.success("✅ Modell geladen (mit Fixes)")
        return processor, model
    except Exception as e:
        st.error(f"Fehler: {e}")
        return None, None

processor, model = load_model()

st.info("Falls das Modell lädt, kannst du hier den Rest einbauen.")
