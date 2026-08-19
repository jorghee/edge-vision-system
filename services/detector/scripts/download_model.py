"""
Initialization script: downloads YOLO models once.
Executed during Docker build, not at runtime.
"""
import os
import requests

MODELS_DIR = "/app/models"
os.makedirs(MODELS_DIR, exist_ok=True)

YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
yolo_path = f"{MODELS_DIR}/yolov8n.pt"

if not os.path.exists(yolo_path):
    print("[SETUP] Downloading yolov8n.pt...")
    r = requests.get(YOLO_URL, stream=True, timeout=120)
    r.raise_for_status()

    with open(yolo_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    print(f"[SETUP] Model saved to {MODELS_DIR}/yolov8n.pt")

# The most used PPE model in open source industrial projects:
# keremberke/hard-hat-detection - YOLOv8 fine-tuned
# Downloaded from the public Hugging Face repository
PPE_HF_URL = (
    "https://huggingface.co/keremberke/yolov8n-hard-hat-detection"
    "/resolve/main/best.pt"
)

ppe_path = f"{MODELS_DIR}/ppe_detector.pt"
print("[SETUP] Downloading PPE model (helmet/vest) from HuggingFace...")

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
                print(f"\r[SETUP] Downloading PPE model... {pct:.1f}%", end="")

    print(f"\n[SETUP] ppe_detector.pt saved ({downloaded/1e6:.1f} MB)")

except Exception as e:
    print(f"[SETUP] Could not download PPE model: {e}")
    print("[SETUP] The system will use color analysis as fallback")
    # Do not abort build — detector.py has a fallback

print("\n[SETUP] Models ready for offline inference.")
