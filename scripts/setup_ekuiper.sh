#!/bin/bash
# Provisions eKuiper with portable plugin, video stream, AI inference rules, and alert routing.
# Idempotent: safe to re-run; deletes existing config before recreating.

set -euo pipefail

EKUIPER_HOST="${EKUIPER_HOST:-localhost}"
EKUIPER_PORT="${EKUIPER_PORT:-9081}"
API_URL="http://${EKUIPER_HOST}:${EKUIPER_PORT}"

# Name of the Docker container running eKuiper
EKUIPER_CONTAINER="${EKUIPER_CONTAINER:-ekuiper-engine}"

echo "[1/6] Cleaning previous configuration..."
for rule in ppe_alert_critical ppe_alert_high ppe_monitor alert_critical alert_high monitor_all; do
    curl -s -X DELETE "${API_URL}/rules/${rule}" > /dev/null 2>&1 || true
done
for stream in video_frames camera_events; do
    curl -s -X DELETE "${API_URL}/streams/${stream}" > /dev/null 2>&1 || true
done
curl -s -X DELETE "${API_URL}/plugins/portables/ppe_inference" > /dev/null 2>&1 || true

echo "[2/6] Registering Portable Plugin (ppe_inference)..."
# eKuiper REST API requires a .zip file, not a directory path.
# Create the zip inside the container using Python (already installed).
docker exec "${EKUIPER_CONTAINER}" python3 -c "
import zipfile, os
plugin_dir = '/kuiper/plugins/portables/ppe_inference'
zip_path = '/tmp/ppe_inference.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir(plugin_dir):
        zf.write(os.path.join(plugin_dir, f), f)
print('Plugin zip created at', zip_path)
"

REGISTER_RESULT=$(curl -s -X POST "${API_URL}/plugins/portables" \
  -H "Content-Type: application/json" \
  -d '{"name": "ppe_inference", "file": "file:///tmp/ppe_inference.zip"}')
echo "  ${REGISTER_RESULT}"

# Wait for the plugin process to start and register functions
sleep 3

echo "[3/6] Creating video stream..."
curl -s -X POST "${API_URL}/streams" \
  -H "Content-Type: application/json" \
  -d '{
    "sql": "CREATE STREAM video_frames () WITH (TYPE=\"video\", CONF_KEY=\"default\", FORMAT=\"binary\")"
  }'
echo ""

echo "[4/6] Creating PPE detection rule (critical alerts)..."
curl -s -X POST "${API_URL}/rules" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "ppe_alert_critical",
    "sql": "SELECT * FROM video_frames WHERE ppeInference(self)->severity = '\''critical'\''",
    "actions": [
      { "mqtt": { "server": "tcp://mqtt:1883", "topic": "edge/alerts", "qos": 1 } },
      { "log": {} }
    ]
  }'
echo ""

echo "[5/6] Creating PPE detection rule (high alerts)..."
curl -s -X POST "${API_URL}/rules" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "ppe_alert_high",
    "sql": "SELECT * FROM video_frames WHERE ppeInference(self)->severity = '\''high'\''",
    "actions": [
      { "mqtt": { "server": "tcp://mqtt:1883", "topic": "edge/alerts", "qos": 1 } },
      { "log": {} }
    ]
  }'
echo ""

echo "[6/6] Creating monitoring rule (all non-clear events)..."
curl -s -X POST "${API_URL}/rules" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "ppe_monitor",
    "sql": "SELECT * FROM video_frames WHERE ppeInference(self)->event_type != '\''clear'\''",
    "actions": [
      { "mqtt": { "server": "tcp://mqtt:1883", "topic": "edge/monitor", "qos": 0 } }
    ]
  }'
echo ""

echo "[OK] eKuiper configuration completed (eKuiper-native pipeline)."
