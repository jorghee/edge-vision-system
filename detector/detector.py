"""
Detector de eventos - Fase 3: Simulado
En fases posteriores este script usará OpenCV para analizar video real.
"""
import json
import time
import random
import os
import logging
from datetime import datetime
import paho.mqtt.client as mqtt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DETECTOR] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)

MQTT_BROKER   = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT     = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC    = os.getenv("MQTT_TOPIC", "camera/events")
CAMERA_ID     = os.getenv("CAMERA_ID", "cam-01")
INTERVAL_SEC  = float(os.getenv("INTERVAL_SEC", "3"))

# Possible events to simulate
POSSIBLE_EVENTS = [
    {"type": "no_helmet",      "confidence": 0.92, "severity": "high"},
    {"type": "person_detected","confidence": 0.87, "severity": "low"},
    {"type": "no_vest",        "confidence": 0.78, "severity": "high"},
    {"type": "clear",          "confidence": 0.99, "severity": "none"},
]

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        log.info(f"[SUCCESS] Conectado al broker MQTT en {MQTT_BROKER}:{MQTT_PORT}")
    else:
        log.error(f"[ERROR] Error de conexión MQTT, código: {rc}")

def on_publish(client, userdata, mid):
    log.debug(f"Mensaje publicado (id={mid})")

def build_event(detected: dict) -> dict:
    """Construye el payload JSON estándar del sistema."""
    return {
        "camera_id":  CAMERA_ID,
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "event_type": detected["type"],
        "confidence": detected["confidence"],
        "severity":   detected["severity"],
        "source":     "simulated",         # Will change to "opencv"
        "metadata": {
            "frame": random.randint(1, 9999),
            "zone":  "entrada-principal"
        }
    }

def main():
    # Configure MQTT client
    client = mqtt.Client(client_id=f"detector-{CAMERA_ID}")
    client.on_connect = on_connect
    client.on_publish = on_publish

    # Try again connection if broker's not ready
    connected = False
    for attempt in range(10):
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            connected = True
            break
        except Exception as e:
            log.warning(f"[WARNNING] Intento {attempt+1}/10: no se pudo conectar - {e}")
            time.sleep(3)

    if not connected:
        log.error("[ERROR] No se pudo conectar al broker MQTT. Abortando.")
        return

    client.loop_start()

    log.info(f"[INFO] Publicando eventos en '{MQTT_TOPIC}' cada {INTERVAL_SEC}s")

    try:
        while True:
            # Select random event (simulate detection)
            detected = random.choice(POSSIBLE_EVENTS)
            event    = build_event(detected)
            payload  = json.dumps(event)

            result = client.publish(MQTT_TOPIC, payload, qos=1)
            log.info(f"[SUCCESS] Evento enviado: {event['event_type']} "
                     f"(severidad={event['severity']}, "
                     f"confianza={event['confidence']})")

            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        log.info("[WARNNING] Detector detenido por el usuario")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
