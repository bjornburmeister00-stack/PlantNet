
---

### **2. README.md für App 2 – PlantNet-300K (Hugging Face)**

```markdown
# 🌳 Pflanzen-Detektor – PlantNet-300K (Hugging Face)

**Schulprojekt 2026**

Fortgeschrittene Pflanzenerkennung mit einem starken vortrainierten KI-Modell.

## Projektbeschreibung
Diese App nutzt ein **Vision Transformer (ViT)** Modell, das auf dem großen **PlantNet-300K** Datensatz trainiert wurde. Es kann eine sehr große Anzahl von Pflanzenarten erkennen und liefert zuverlässige Ergebnisse mit Wahrscheinlichkeitsangabe sowie Wikipedia-Links.

## Features
- Foto-Upload und Live-Kamera
- Hochgenaue Bildklassifizierung
- Anzeige der Top-Ergebnisse mit Prozentwerten
- Automatische Wikipedia-Links zu den erkannten Arten
- Einheitliches grün-weißes Design (passend zu App 1)

## Erkannte Pflanzen
Das Modell kann **über 1000 verschiedene Pflanzenarten** weltweit erkennen.  
Besonders gut erkennt es unter anderem:

- Viele heimische Bäume (z. B. Birke, Eichen, Fichten, Kiefern)
- Häufige Wiesen- und Gartenblumen (z. B. Gänseblümchen, Klatschmohn, Lavendel, Sonnenblume)
- Zahlreiche Kräuter, Stauden und exotische Pflanzen
- Farne, Kakteen und Wasserpflanzen

Die genaue Liste umfasst über 1000 Arten aus dem PlantNet-300K Datensatz.

## Technologien
- Streamlit
- Hugging Face Transformers
- Vision Transformer (ViT)
- PlantNet-300K Modell

## Installation lokal
```bash
pip install -r requirements.txt
streamlit run app.py
