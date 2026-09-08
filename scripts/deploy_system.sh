#!/bin/bash
# Master orchestrator script to deploy the full Edge Vision System.
# Deploys the Central Server locally, and the Edge components to a remote Raspberry Pi.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "========================================================="
echo "      Edge Vision System - Full System Deployment"
echo "========================================================="
echo "[INFO] This script will configure THIS machine as the Central Server"
echo "       and deploy the AI components to a remote Raspberry Pi."
echo "========================================================="
echo ""

# Start the Central Server locally
echo "[1/4] Starting Central Server (MQTT Broker + Action Service)..."
docker compose -f "${PROJECT_ROOT}/docker-compose.server.yml" up --build -d

# Determine Central Server IP
echo ""
echo "[2/4] Detecting local IP address for Edge device to connect to..."
# Use hostname -I to get LAN IPs, pick the first one (usually the primary network interface)
if command -v hostname >/dev/null 2>&1; then
    DETECTED_IP=$(hostname -I | awk '{print $1}')
else
    # Fallback for some systems (like Mac if running there)
    DETECTED_IP=$(ip route get 1.1.1.1 | awk '{print $7}')
fi

read -rp "Is this the correct Central Server IP? [${DETECTED_IP}]: " USER_IP
SERVER_IP="${USER_IP:-${DETECTED_IP}}"

if [ -z "${SERVER_IP}" ]; then
    echo "[ERROR] Could not determine a valid IP address. Deployment aborted."
    exit 1
fi

MQTT_URL="tcp://${SERVER_IP}:1883"
echo "  -> Edge device will be configured to connect to: ${MQTT_URL}"

# Deploy to Edge (Raspberry Pi)
echo ""
echo "[3/4] Initiating Edge deployment..."
# Call the edge deployment script with the MQTT URL as an argument
bash "${SCRIPT_DIR}/deploy_edge.sh" "${MQTT_URL}"

# Final Verification Instructions
echo ""
echo "[4/4] Deployment orchestration complete!"
echo "========================================================="
echo "The Central Server is running locally."
echo "The Edge (RPi) should now be capturing video and sending alerts."
echo ""
echo "To verify alerts on this Central Server, run:"
echo "  docker exec mqtt-broker mosquitto_sub -t 'edge/#' -v -u edge_device -P 'SecureEdge2026!'"
echo "========================================================="
