"""
Action Service
Listens for alerts on 'edge/alerts' and executes the corresponding actions.
In production it could: send email, call an API, activate a relay, etc.
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

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
ALERT_TOPIC = os.getenv("ALERT_TOPIC", "edge/alerts")
ACTION_TOPIC = os.getenv("ACTION_TOPIC", "edge/actions")

# Available actions


def action_log_alert(alert: dict, client: mqtt.Client):
    """Logs the alert with a highlighted format."""
    log.warning(
        f"[INFO] ALERT DETECTED\n"
        f"   Camera   : {alert.get('camera_id', 'N/A')}\n"
        f"   Event    : {alert.get('event_type', 'N/A')}\n"
        f"   Severity : {alert.get('severity', 'N/A')}\n"
        f"   Confidence: {alert.get('confidence', 'N/A')}\n"
        f"   Time     : {alert.get('timestamp', 'N/A')}"
    )


def action_publish_response(alert: dict, client: mqtt.Client):
    """Publishes a response to the actions topic."""
    response = {
        "action": "alert_triggered",
        "source_event": alert.get("event_type"),
        "camera_id": alert.get("camera_id"),
        "handled_at": datetime.utcnow().isoformat() + "Z",
        "message": f"Alert processed: {alert.get('event_type')} detected",
        "recommended_action": get_recommendation(alert.get("event_type", ""))
    }
    client.publish(ACTION_TOPIC, json.dumps(response), qos=1)
    log.info(f"[SUCCESS] Response published to '{ACTION_TOPIC}'")


def get_recommendation(event_type: str) -> str:
    """Returns a recommendation based on the event type."""
    recommendations = {
        "no_helmet":  "Stop operation. Require safety helmet.",
        "no_vest":    "Warn the operator. Require reflective vest.",
        "intrusion":  "Activate alarm. Notify security.",
    }
    return recommendations.get(event_type, "Review camera manually.")

# MQTT Callbacks


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        log.info(f"[SUCCESS] Connected to MQTT broker at {
                 MQTT_BROKER}:{MQTT_PORT}")
        # Subscribe to alerts topic
        client.subscribe(ALERT_TOPIC, qos=1)
        log.info(f"[LISTEN] Listening for alerts on '{ALERT_TOPIC}'")
    else:
        log.error(f"[ERROR] MQTT connection error, code: {rc}")


def on_message(client, userdata, msg):
    """Executed every time a message arrives."""
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        log.info(f"[INFO] Alert received from '{msg.topic}'")

        # Execute actions
        action_log_alert(payload, client)
        action_publish_response(payload, client)

    except json.JSONDecodeError as e:
        log.error(f"Error parsing message: {e}")
    except Exception as e:
        log.error(f"Error processing alert: {e}")


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
            log.warning(f"Attempt {attempt+1}/10: {e}")
            time.sleep(3)

    log.info("[SUCCESS] Action service started. Waiting for alerts...")

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        log.info("[WARNING] Service stopped")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
