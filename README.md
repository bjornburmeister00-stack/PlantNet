# 🌳 Pflanzen-Detektor – PlantNet-300K (Hugging Face)

**Schulprojekt 2026**

Fortgeschrittene KI-gestützte Pflanzenerkennung mit einem vortrainierten Modell.

## Projektbeschreibung
Diese Web-App nutzt ein **Vision Transformer (ViT)** Modell von Hugging Face, das auf dem großen **PlantNet-300K** Datensatz trainiert wurde. Es kann über 1000 verschiedene Pflanzenarten erkennen und liefert zu jedem Ergebnis einen direkten Wikipedia-Link.

## Features
- Foto-Upload und Live-Kamera
- Schnelle und genaue Bilderkennung
- Anzeige der Top-Ergebnisse mit prozentualer Sicherheit
- Automatische Wikipedia-Links zu den erkannten Pflanzen
- Einheitliches grün-weißes Design (passend zur Teachable Machine App)

## Erkannte Pflanzen
Das Modell erkennt **über 1000 verschiedene Pflanzenarten** weltweit.  
Dazu gehören unter anderem viele heimische Bäume (z. B. Birke, Rotbuche, Eichen, Fichten, Kiefern), Wiesenblumen (z. B. Gänseblümchen, Klatschmohn, Sonnenblume, Lavendel) sowie zahlreiche exotische und seltene Arten.

## Technologien
- Streamlit
- Hugging Face Transformers
- Vision Transformer (ViT)
- PlantNet-300K Modell

## Installation lokal

```bash
pip install -r requirements.txt
streamlit run app.py
