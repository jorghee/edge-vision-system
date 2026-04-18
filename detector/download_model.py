"""
Script de inicialización: descarga los modelos YOLO una sola vez.
Se ejecuta durante el build de Docker, no en runtime.
"""
import os
from ultralytics import YOLO

MODELS_DIR = "/app/models"
os.makedirs(MODELS_DIR, exist_ok=True)

print("[SETUP] Descargando YOLOv8n base (detección de personas)...")
model = YOLO("yolov8n.pt")
model.save(f"{MODELS_DIR}/yolov8n.pt")
print(f"[SETUP] Modelo guardado en {MODELS_DIR}/yolov8n.pt")

print("[SETUP] Modelos listos.")
