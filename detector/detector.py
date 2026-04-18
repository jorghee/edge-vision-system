import json
import time
import random
import os
import logging
import numpy as np
from datetime import datetime
import paho.mqtt.client as mqtt

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DETECTOR] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)

MQTT_BROKER   = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT     = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC    = os.getenv("MQTT_TOPIC", "camera/events")
CAMERA_ID     = os.getenv("CAMERA_ID", "cam-01")
CAMERA_INDEX     = int(os.getenv("CAMERA_INDEX", "0"))
INTERVAL_SEC  = float(os.getenv("INTERVAL_SEC", "3"))
USE_SIMULATION   = os.getenv("USE_SIMULATION", "false").lower() == "true"

class HelmetDetector:
    """
    Detector simplificado basado en análisis de color.
    En producción se usaría un modelo YOLO o similar.

    Si detecta una región anaranjada/amarilla en la parte
    superior del frame, asume que hay un casco de seguridad visible.
    """

    def analyze_frame(self, frame: np.ndarray) -> dict:
        height, width = frame.shape[:2]

        upper_region = frame[0:height//2, :]

        hsv = cv2.cvtColor(upper_region, cv2.COLOR_BGR2HSV)

        # Color range
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow  = cv2.inRange(hsv, lower_yellow, upper_yellow)

        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([15, 255, 255])
        mask_orange  = cv2.inRange(hsv, lower_orange, upper_orange)

        # Masks
        mask_combined = cv2.bitwise_or(mask_yellow, mask_orange)
        helmet_pixels = cv2.countNonZero(mask_combined)
        total_pixels  = upper_region.shape[0] * upper_region.shape[1]
        helmet_ratio  = helmet_pixels / total_pixels

        # Threshold
        helmet_detected = helmet_ratio > 0.05
        confidence = min(0.99, helmet_ratio * 10)  # Normalizar

        return {
            "helmet_detected": helmet_detected,
            "confidence": round(confidence, 2),
            "helmet_ratio": round(helmet_ratio, 4)
        }

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        log.info(f"[SUCCESS] Conectado al broker MQTT en {MQTT_BROKER}:{MQTT_PORT}")
    else:
        log.error(f"[ERROR] Error de conexión MQTT, código: {rc}")

def build_event(event_type: str, confidence: float, source: str, frame_num: int) -> dict:
    severity_map = {
        "no_helmet":       "high",
        "helmet_detected": "none",
        "no_vest":         "high",
        "person_detected": "low",
        "clear":           "none",
    }
    return {
        "camera_id":  CAMERA_ID,
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "confidence": confidence,
        "severity":   severity_map.get(event_type, "medium"),
        "source":     source,
        "metadata": {
            "frame": frame_num,
            "zone": "entrada-principal"
        }
    }

def run_opencv_detector(client: mqtt.Client):
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        log.error(f"[ERROR] No se pudo abrir la cámara (índice={CAMERA_INDEX})")
        log.info("[INFO] Revisa que la cámara esté conectada y no usada por otro proceso")
        return

    log.info(f"[SUCCESS] Cámara abierta (índice={CAMERA_INDEX})")
    detector  = HelmetDetector()
    frame_num = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("[WARNING] No se pudo leer frame. Reintentando...")
                time.sleep(1)
                continue

            frame_num += 1

            # Analyze every N seconds para no saturar MQTT
            if frame_num % int(30 * INTERVAL_SEC) == 0:
                result = detector.analyze_frame(frame)

                if result["helmet_detected"]:
                    event_type = "helmet_detected"
                else:
                    event_type = "no_helmet"

                event   = build_event(event_type, result["confidence"], "opencv", frame_num)
                payload = json.dumps(event)
                client.publish(MQTT_TOPIC, payload, qos=1)
                log.info(f"[INFO] {event_type} | confianza={result['confidence']} | frame={frame_num}")

    except KeyboardInterrupt:
        log.info("[INFO] Detector detenido")
    finally:
        cap.release()
        log.info("[INFO] Cámara liberada")

def run_simulation(client: mqtt.Client):
    import random
    EVENTS = [
        ("no_helmet", 0.92),
        ("person_detected", 0.87),
        ("no_vest", 0.78),
        ("clear", 0.99),
    ]
    frame_num = 0
    log.info("[SIM] Modo simulación activo")
    while True:
        frame_num += 1
        event_type, confidence = random.choice(EVENTS)
        event   = build_event(event_type, confidence, "simulated", frame_num)
        payload = json.dumps(event)
        client.publish(MQTT_TOPIC, payload, qos=1)
        log.info(f"[SIM] {event_type} | conf={confidence}")
        time.sleep(INTERVAL_SEC)

def main():
    # Configure MQTT client
    client = mqtt.Client(client_id=f"detector-{CAMERA_ID}")
    client.on_connect = on_connect

    for attempt in range(10):
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            break
        except Exception as e:
            log.warning(f"[WARNING] Intento {attempt+1}/10: {e}")
            time.sleep(3)

    client.loop_start()

    if USE_SIMULATION or not OPENCV_AVAILABLE:
        run_simulation(client)
    else:
        run_opencv_detector(client)

    client.loop_stop()
    client.disconnect()

if __name__ == "__main__":
    main()
