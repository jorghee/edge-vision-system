"""
Servicio de Acciones
Escucha alertas en 'edge/alerts' y ejecuta las acciones correspondientes.
En producción podría: enviar email, llamar una API, activar un relay, etc.
"""

import json
import time
import os
import logging
from datetime import datetime
import paho.mqtt.client as mqtt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ACTION] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)

MQTT_BROKER      = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT        = int(os.getenv("MQTT_PORT", "1883"))
ALERT_TOPIC      = os.getenv("ALERT_TOPIC", "edge/alerts")
ACTION_TOPIC     = os.getenv("ACTION_TOPIC", "edge/actions")

# Available actions
def action_log_alert(alert: dict, client: mqtt.Client):
    """Loguea la alerta con formato destacado."""
    log.warning(
        f"[INFO] ALERTA DETECTADA\n"
        f"   Cámara   : {alert.get('camera_id', 'N/A')}\n"
        f"   Evento   : {alert.get('event_type', 'N/A')}\n"
        f"   Severidad: {alert.get('severity', 'N/A')}\n"
        f"   Confianza: {alert.get('confidence', 'N/A')}\n"
        f"   Tiempo   : {alert.get('timestamp', 'N/A')}"
    )

def action_publish_response(alert: dict, client: mqtt.Client):
    """Publica una respuesta en el topic de acciones."""
    response = {
        "action": "alert_triggered",
        "source_event": alert.get("event_type"),
        "camera_id": alert.get("camera_id"),
        "handled_at": datetime.utcnow().isoformat() + "Z",
        "message": f"Alerta procesada: {alert.get('event_type')} detectado",
        "recommended_action": get_recommendation(alert.get("event_type", ""))
    }
    client.publish(ACTION_TOPIC, json.dumps(response), qos=1)
    log.info(f"[SUCCESS] Respuesta publicada en '{ACTION_TOPIC}'")

def get_recommendation(event_type: str) -> str:
    """Retorna una recomendación según el tipo de evento."""
    recommendations = {
        "no_helmet":  "Detener operación. Exigir uso de casco de seguridad.",
        "no_vest":    "Advertir al operario. Exigir chaleco reflectivo.",
        "intrusion":  "Activar alarma. Notificar seguridad.",
    }
    return recommendations.get(event_type, "Revisar cámara manualmente.")

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        log.info(f"[SUCCESS] Conectado al broker MQTT en {MQTT_BROKER}:{MQTT_PORT}")
        # Suscribirse al topic de alertas
        client.subscribe(ALERT_TOPIC, qos=1)
        log.info(f"[LISTEN] Escuchando alertas en '{ALERT_TOPIC}'")
    else:
        log.error(f"[ERROR] Error de conexión MQTT, código: {rc}")

def on_message(client, userdata, msg):
    """Se ejecuta cada vez que llega un mensaje."""
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        log.info(f"[INFO] Alerta recibida desde '{msg.topic}'")

        # Ejecutar acciones
        action_log_alert(payload, client)
        action_publish_response(payload, client)

    except json.JSONDecodeError as e:
        log.error(f"Error al parsear mensaje: {e}")
    except Exception as e:
        log.error(f"Error procesando alerta: {e}")

def main():
    client = mqtt.Client(client_id="action-service")
    client.on_connect = on_connect
    client.on_message = on_message

    # Try again connection
    for attempt in range(10):
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            break
        except Exception as e:
            log.warning(f"Intento {attempt+1}/10: {e}")
            time.sleep(3)

    log.info("[SUCCESS] Servicio de acciones iniciado. Esperando alertas...")

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        log.info("[WARNING] Servicio detenido")
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()
