#!/bin/bash
# Provisions eKuiper with streams and SQL filtering rules.
# Idempotent: safe to re-run; deletes existing config before recreating.

set -euo pipefail

EKUIPER_HOST="${EKUIPER_HOST:-localhost}"
EKUIPER_PORT="${EKUIPER_PORT:-9081}"
API_URL="http://${EKUIPER_HOST}:${EKUIPER_PORT}"

echo "[1/4] Cleaning previous configuration..."
curl -s -X DELETE "${API_URL}/rules/alert_critical" > /dev/null
curl -s -X DELETE "${API_URL}/rules/alert_high" > /dev/null
curl -s -X DELETE "${API_URL}/rules/monitor_all" > /dev/null
curl -s -X DELETE "${API_URL}/streams/camera_events" > /dev/null

echo "[2/4] Creating stream 'camera_events'..."
curl -s -X POST "${API_URL}/streams" \
  -H "Content-Type: application/json" \
  -d '{
    "sql": "CREATE STREAM camera_events() WITH (DATASOURCE=\"camera/events\", FORMAT=\"json\", TYPE=\"mqtt\", CONF_KEY=\"default\")"
  }' > /dev/null

echo "[3/4] Creating rule 'alert_critical'..."
curl -s -X POST "${API_URL}/rules" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "alert_critical",
    "sql": "SELECT camera_id, event_type, severity, confidence, timestamp, metadata FROM camera_events WHERE severity = '\''critical'\''",
    "actions": [
      { "mqtt": { "server": "tcp://mqtt:1883", "topic": "edge/alerts", "qos": 1 } },
      { "log": {} }
    ]
  }' > /dev/null

echo "[4/4] Creating rules 'alert_high' and 'monitor_all'..."
curl -s -X POST "${API_URL}/rules" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "alert_high",
    "sql": "SELECT camera_id, event_type, severity, confidence, timestamp, metadata FROM camera_events WHERE severity = '\''high'\''",
    "actions": [
      { "mqtt": { "server": "tcp://mqtt:1883", "topic": "edge/alerts", "qos": 1 } },
      { "log": {} }
    ]
  }' > /dev/null

curl -s -X POST "${API_URL}/rules" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "monitor_all",
    "sql": "SELECT camera_id, event_type, severity, confidence, timestamp FROM camera_events WHERE event_type != '\''clear'\''",
    "actions": [
      { "mqtt": { "server": "tcp://mqtt:1883", "topic": "edge/monitor", "qos": 0 } }
    ]
  }' > /dev/null

echo "[OK] eKuiper configuration completed."
