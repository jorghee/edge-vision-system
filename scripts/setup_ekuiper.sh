#!/bin/bash
# Provisions eKuiper with video stream, AI inference rules, and alert routing.
# Idempotent: safe to re-run; deletes existing config before recreating.

set -euo pipefail

EKUIPER_HOST="${EKUIPER_HOST:-localhost}"
EKUIPER_PORT="${EKUIPER_PORT:-9081}"
API_URL="http://${EKUIPER_HOST}:${EKUIPER_PORT}"

echo "[1/5] Cleaning previous configuration..."
for rule in ppe_alert_critical ppe_alert_high ppe_monitor alert_critical alert_high monitor_all; do
    curl -s -X DELETE "${API_URL}/rules/${rule}" > /dev/null 2>&1 || true
done
for stream in video_frames camera_events; do
    curl -s -X DELETE "${API_URL}/streams/${stream}" > /dev/null 2>&1 || true
done

echo "[2/5] Creating video stream..."
curl -s -X POST "${API_URL}/streams" \
  -H "Content-Type: application/json" \
  -d '{
    "sql": "CREATE STREAM video_frames () WITH (TYPE=\"video\", CONF_KEY=\"default\", FORMAT=\"binary\")"
  }'

echo "[3/5] Creating PPE detection rule (critical alerts)..."
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

echo "[4/5] Creating PPE detection rule (high alerts)..."
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

echo "[5/5] Creating monitoring rule (all non-clear events)..."
curl -s -X POST "${API_URL}/rules" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "ppe_monitor",
    "sql": "SELECT * FROM video_frames WHERE ppeInference(self)->event_type != '\''clear'\''",
    "actions": [
      { "mqtt": { "server": "tcp://mqtt:1883", "topic": "edge/monitor", "qos": 0 } }
    ]
  }'

echo "[OK] eKuiper configuration completed (eKuiper-native pipeline)."
