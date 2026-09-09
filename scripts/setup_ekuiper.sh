#!/bin/bash
# Provisions eKuiper with portable plugin, camera stream, AI inference rules, and alert routing.
# Idempotent: safe to re-run; deletes existing config before recreating.
#
# Architecture:
#   RPi Camera → MediaMTX (RTSP) → eKuiper → ppeInference() → MQTT → Central Server
#
# Environment variables (read from the eKuiper container or overridden):
#   MQTT_SERVER_URL  - Central server MQTT (default from container env)
#   MQTT_USERNAME    - MQTT auth username (default from container env)
#   MQTT_PASSWORD    - MQTT auth password (default from container env)

set -euo pipefail

EKUIPER_HOST="${EKUIPER_HOST:-localhost}"
EKUIPER_PORT="${EKUIPER_PORT:-9081}"
API_URL="http://${EKUIPER_HOST}:${EKUIPER_PORT}"

# Name of the Docker container running eKuiper
EKUIPER_CONTAINER="${EKUIPER_CONTAINER:-ekuiper-engine}"

# Read MQTT config from the eKuiper container's environment (or override locally)
MQTT_SERVER_URL="${MQTT_SERVER_URL:-$(docker exec "${EKUIPER_CONTAINER}" printenv MQTT_SERVER_URL 2>/dev/null || echo 'tcp://localhost:1883')}"
MQTT_USERNAME="${MQTT_USERNAME:-$(docker exec "${EKUIPER_CONTAINER}" printenv MQTT_USERNAME 2>/dev/null || echo '')}"
MQTT_PASSWORD="${MQTT_PASSWORD:-$(docker exec "${EKUIPER_CONTAINER}" printenv MQTT_PASSWORD 2>/dev/null || echo '')}"

echo "MQTT target: ${MQTT_SERVER_URL} (user: ${MQTT_USERNAME:-anonymous})"

# ── Helper: create an eKuiper rule with retry logic ──────────────────────────
# On ARM64, the portable plugin (Python + OpenCV + YOLO) can take over 60s to
# fully initialize. eKuiper validates the SQL by calling the function at rule
# creation time, which may time out if the plugin is still loading.
# This function retries with exponential backoff until the plugin is ready.
create_rule() {
    local rule_name="$1"
    local rule_json="$2"
    local max_attempts=6
    local wait_time=20

    for attempt in $(seq 1 $max_attempts); do
        result=$(curl -s -X POST "${API_URL}/rules" \
          -H "Content-Type: application/json" \
          -d "$rule_json")

        if echo "$result" | grep -q '"error"'; then
            echo "  Attempt ${attempt}/${max_attempts} — plugin not ready yet."
            if [ $attempt -lt $max_attempts ]; then
                echo "  Waiting ${wait_time}s for plugin initialization..."
                sleep $wait_time
                wait_time=$((wait_time + 15))
            else
                echo "  [ERROR] Rule '${rule_name}' failed after ${max_attempts} attempts."
                echo "  Last error: ${result}"
                return 1
            fi
        else
            echo "  Rule '${rule_name}' created successfully."
            return 0
        fi
    done
}

echo "[1/6] Cleaning previous configuration..."
for rule in ppe_alert_critical ppe_alert_high ppe_monitor alert_critical alert_high monitor_all; do
    curl -s -X DELETE "${API_URL}/rules/${rule}" > /dev/null 2>&1 || true
done
for stream in camera_frames video_frames camera_events; do
    curl -s -X DELETE "${API_URL}/streams/${stream}" > /dev/null 2>&1 || true
done
curl -s -X DELETE "${API_URL}/plugins/portables/ppe_inference" > /dev/null 2>&1 || true

echo "[2/6] Registering Portable Plugin (ppe_inference)..."
docker exec "${EKUIPER_CONTAINER}" python3 -c "
import zipfile, os
plugin_dir = '/kuiper/plugins/portables/ppe_inference'
zip_path = '/tmp/ppe_inference.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(plugin_dir):
        for f in files:
            full = os.path.join(root, f)
            arc = os.path.relpath(full, plugin_dir)
            zf.write(full, arc)
print('Plugin zip created at', zip_path)
"

REGISTER_RESULT=$(curl -s -X POST "${API_URL}/plugins/portables" \
  -H "Content-Type: application/json" \
  -d '{"name": "ppe_inference", "file": "file:///tmp/ppe_inference.zip"}')
echo "  ${REGISTER_RESULT}"

echo "[3/6] Creating camera stream (portable source: cameraSource)..."
curl -s -X POST "${API_URL}/streams" \
  -H "Content-Type: application/json" \
  -d '{
    "sql": "CREATE STREAM camera_frames () WITH (TYPE=\"cameraSource\", CONF_KEY=\"default\", FORMAT=\"json\")"
  }'
echo ""

# Build MQTT sink config with authentication
MQTT_SINK_ALERTS="{\"server\":\"${MQTT_SERVER_URL}\",\"topic\":\"edge/alerts\",\"qos\":1,\"username\":\"${MQTT_USERNAME}\",\"password\":\"${MQTT_PASSWORD}\",\"sendSingle\":true,\"maxDiskCache\":10000,\"bufferPageSize\":1,\"resendInterval\":2000,\"cleanCacheAtStop\":false}"
MQTT_SINK_MONITOR="{\"server\":\"${MQTT_SERVER_URL}\",\"topic\":\"edge/monitor\",\"qos\":0,\"username\":\"${MQTT_USERNAME}\",\"password\":\"${MQTT_PASSWORD}\",\"sendSingle\":true,\"maxDiskCache\":10000,\"bufferPageSize\":1,\"resendInterval\":2000,\"cleanCacheAtStop\":false}"

echo "[4/6] Creating PPE detection rule (critical alerts)..."
create_rule "ppe_alert_critical" "{
    \"id\": \"ppe_alert_critical\",
    \"sql\": \"SELECT ppeInference(frame) as detection FROM camera_frames WHERE ppeInference(frame)->severity = 'critical'\",
    \"actions\": [
      {\"mqtt\": ${MQTT_SINK_ALERTS}},
      {\"log\": {}}
    ]
  }"

echo "[5/6] Creating PPE detection rule (high alerts)..."
create_rule "ppe_alert_high" "{
    \"id\": \"ppe_alert_high\",
    \"sql\": \"SELECT ppeInference(frame) as detection FROM camera_frames WHERE ppeInference(frame)->severity = 'high'\",
    \"actions\": [
      {\"mqtt\": ${MQTT_SINK_ALERTS}},
      {\"log\": {}}
    ]
  }"

echo "[6/6] Creating monitoring rule (all non-clear events)..."
create_rule "ppe_monitor" "{
    \"id\": \"ppe_monitor\",
    \"sql\": \"SELECT ppeInference(frame) as detection FROM camera_frames WHERE ppeInference(frame)->event_type != 'clear'\",
    \"actions\": [
      {\"mqtt\": ${MQTT_SINK_MONITOR}}
    ]
  }"

echo ""
echo "[OK] eKuiper configuration completed."
echo ""
echo "MQTT sink target: ${MQTT_SERVER_URL}"
echo ""
echo "Verify with:"
echo "  curl -s http://localhost:9081/rules/ppe_monitor/status | python3 -m json.tool"
