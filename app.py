import streamlit as st
from PIL import Image
from transformers import pipeline

# ====================== DESIGN ======================
st.set_page_config(
    page_title="Pflanzen-Detektor",
    page_icon="🌳",
    layout="wide"
)

st.markdown("""
<style>
    .main {background-color: #f0f8f0;}
    h1 {color: #228B22; text-align: center;}
    .result-box {
        background-color: #1e3a2f; 
        color: white;
        padding: 1.6em; 
        border-radius: 12px; 
        margin: 1em 0;
    }
    .footer {text-align: center; color: #555; margin-top: 4em;}
</style>
""", unsafe_allow_html=True)

st.title("🌳 Pflanzen-Detektor")
st.markdown("**Plant Identification Modell (Hugging Face)**")

# ====================== MODELL ======================
@st.cache_resource
def load_classifier():
    return pipeline("image-classification", model="umutbozdag/plant-identity")

classifier = load_classifier()

# ====================== UPLOAD ======================
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Bild hochladen (JPG/PNG)", type=["jpg", "jpeg", "png"])
with col2:
    camera_file = st.camera_input("Live-Kamera")

input_image = None
if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
elif camera_file is not None:
    input_image = Image.open(camera_file)

if input_image is not None:
    st.image(input_image, caption="Dein Bild", use_column_width=True)

    with st.spinner("Analysiere Bild..."):
        results = classifier(input_image)

    # Ergebnisse anzeigen
    st.subheader("🔍 Erkennungsergebnisse")
    
    for i, result in enumerate(results[:5]):  # Top 5 Ergebnisse
        label = result['label']
        score = result['score'] * 100
        
        st.markdown(f"""
        <div class="result-box">
            <strong>{label}</strong><br>
            <span style="color:#90EE90">Sicherheit: {score:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

# ====================== INFO ======================
st.markdown("---")
st.info("""
**Hinweis:**  
Dieses Modell erkennt viele verschiedene Pflanzen.  
Die Ergebnisse sind meist wissenschaftliche oder englische Namen.
""")

st.markdown('<p class="footer">Schulprojekt 2026 – [Dein Name]</p>', unsafe_allow_html=True)
