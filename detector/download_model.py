"""
Script de inicialización: descarga los modelos YOLO una sola vez.
Se ejecuta durante el build de Docker, no en runtime.
"""
import os
import requests

MODELS_DIR = "/app/models"
os.makedirs(MODELS_DIR, exist_ok=True)

YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
yolo_path = f"{MODELS_DIR}/yolov8n.pt"

if not os.path.exists(yolo_path):
  print("[SETUP] Descargando yolov8n.pt...")
  r = requests.get(YOLO_URL, stream=True, timeout=120)
  r.raise_for_status()

  with open(yolo_path, "wb") as f:
      for chunk in r.iter_content(8192):
          f.write(chunk)
  print(f"[SETUP] Modelo guardado en {MODELS_DIR}/yolov8n.pt")

# El modelo EPP más usado en proyectos industriales open source:
# keremberke/hard-hat-detection - YOLOv8 fine-tuned
# Lo descargamos desde el repositorio público de Hugging Face
PPE_HF_URL = (
    "https://huggingface.co/keremberke/yolov8n-hard-hat-detection"
    "/resolve/main/best.pt"
)

ppe_path = f"{MODELS_DIR}/ppe_detector.pt"
print("[SETUP] Descargando modelo EPP (casco/chaleco) desde HuggingFace...")

try:
    response = requests.get(PPE_HF_URL, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(ppe_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r[SETUP] Descargando EPP model... {pct:.1f}%", end="")

    print(f"\n[SETUP] ppe_detector.pt guardado ({downloaded/1e6:.1f} MB)")

except Exception as e:
    print(f"[SETUP] No se pudo descargar modelo EPP: {e}")
    print("[SETUP] El sistema usará análisis de color como fallback")
    # No abortar el build — el detector.py tiene fallback

print("\n[SETUP] Modelos listos para inferencia offline.")
