#!/bin/bash
# Automates the deployment of the Edge Vision System to a Raspberry Pi.
# eKuiper-native architecture: all services run in Docker on the RPi.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "========================================================="
echo "        Edge Vision System - Raspberry Pi Deploy"
echo "========================================================="
echo "[INFO] Both devices must be on the same network."
echo "[INFO] The Raspberry Pi must be powered on with SSH enabled."
echo "========================================================="
echo ""

MQTT_SERVER_URL="${1:-}"
if [ -z "${MQTT_SERVER_URL}" ]; then
    echo "[ERROR] MQTT_SERVER_URL is required."
    echo "Usage: bash scripts/deploy_edge.sh <MQTT_SERVER_URL>"
    echo "Example: bash scripts/deploy_edge.sh tcp://192.168.1.100:1883"
    exit 1
fi
echo "[INFO] Edge will connect to central server at: ${MQTT_SERVER_URL}"
echo ""

# Prepare models locally (TFLite + NCNN)
echo "[1/6] Preparing models locally..."
bash "${SCRIPT_DIR}/prepare_models.sh"

# Push local commits so the RPi can clone the latest code
echo ""
echo "[2/6] Pushing local commits to remote..."
if ! git -C "${PROJECT_ROOT}" diff --quiet HEAD 2>/dev/null; then
    echo "[WARN] You have uncommitted changes. They will NOT be available on the RPi."
fi
git -C "${PROJECT_ROOT}" push || {
    echo "[ERROR] git push failed. Ensure your remote is configured and you have access."
    exit 1
}

# Prompt for RPi credentials
echo ""
echo "[3/6] SSH Connection Details"
read -rp "Raspberry Pi username [pi]: " RPI_USER
RPI_USER="${RPI_USER:-pi}"
read -rp "Raspberry Pi IP address: " RPI_IP

if [ -z "${RPI_IP}" ]; then
    echo "[ERROR] IP address cannot be empty."
    exit 1
fi

# Verify connectivity
echo ""
echo "[4/6] Verifying connectivity to ${RPI_USER}@${RPI_IP}..."
if ! ping -c 1 -W 3 "${RPI_IP}" >/dev/null 2>&1; then
    echo "[ERROR] Host ${RPI_IP} is unreachable. Check your network."
    exit 1
fi
echo "[OK] Host reachable."
echo "[INFO] To avoid repeated SSH prompts: ssh-copy-id ${RPI_USER}@${RPI_IP}"

# Install system dependencies on RPi (only Docker and Git needed now)
echo ""
echo "[5/6] Installing system dependencies on Raspberry Pi..."
ssh -t "${RPI_USER}@${RPI_IP}" << 'REMOTE_DEPS'
    set -euo pipefail

    echo "  Updating package lists..."
    sudo apt-get update -qq

    echo "  Installing Git and curl..."
    sudo apt-get install -y -qq git curl

    if ! command -v docker >/dev/null 2>&1; then
        echo "  Installing Docker..."
        curl -fsSL https://get.docker.com | sh
        sudo usermod -aG docker "$USER"
        echo "  Docker installed. Group 'docker' added to user."
    else
        echo "  Docker is already installed."
    fi
REMOTE_DEPS

# Clone or update repository, then transfer models
echo ""
echo "[6/6] Syncing repository and transferring models..."

REPO_URL=$(git -C "${PROJECT_ROOT}" config --get remote.origin.url || echo "")
if [ -z "${REPO_URL}" ]; then
    echo "[ERROR] Could not determine Git remote URL."
    exit 1
fi
# git@github.com:user/repo.git to https://github.com/user/repo.git
HTTPS_URL=$(echo "${REPO_URL}" | sed -E 's|^git@([^:]+):|https://\1/|')

RPI_PROJECT_DIR="edge-vision-system"

ssh "${RPI_USER}@${RPI_IP}" "
    if [ ! -d '${RPI_PROJECT_DIR}' ]; then
        echo '  Cloning repository (branch: feature/ekuiper)...'
        git clone -b feature/ekuiper '${HTTPS_URL}' '${RPI_PROJECT_DIR}'
    else
        echo '  Updating repository (branch: feature/ekuiper)...'
        cd '${RPI_PROJECT_DIR}' && git fetch && git checkout feature/ekuiper && git pull origin feature/ekuiper
    fi
"

# Transfer pre-exported models (not in git due to .gitignore)
MODELS_DIR="${PROJECT_ROOT}/services/detector/models"
RPI_MODELS_DIR="${RPI_PROJECT_DIR}/services/detector/models"

ssh "${RPI_USER}@${RPI_IP}" "mkdir -p '${RPI_MODELS_DIR}'"
echo "  Transferring models (TFLite + NCNN)..."
scp -r "${MODELS_DIR}/"* "${RPI_USER}@${RPI_IP}:${RPI_MODELS_DIR}/"

# Start the system on RPi
echo ""
echo "[OK] Starting Edge Vision System on Raspberry Pi..."
ssh -t "${RPI_USER}@${RPI_IP}" "
    cd '${RPI_PROJECT_DIR}'
    if groups | grep -q docker; then
        MQTT_SERVER_URL='${MQTT_SERVER_URL}' bash scripts/start_rpi.sh
    else
        echo '[INFO] Running with newgrp docker for first-time setup...'
        sg docker -c 'MQTT_SERVER_URL=\"${MQTT_SERVER_URL}\" bash scripts/start_rpi.sh'
    fi
"
