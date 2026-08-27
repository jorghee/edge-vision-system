#!/bin/bash
# Automates the deployment of the Edge Vision System to a Raspberry Pi.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "========================================================="
echo "        Edge Vision System - Raspberry Pi Deploy"
echo "========================================================="
echo "[INFO] Ensure your laptop and the Raspberry Pi are on the same network."
echo "[INFO] Ensure the Raspberry Pi is powered on and accessible via SSH."
echo "========================================================="
echo ""

echo "[1/6] Preparing models locally..."
bash "${SCRIPT_DIR}/prepare_models.sh"

echo ""
echo "[2/6] SSH Connection Details"
read -p "Enter Raspberry Pi username (e.g., pi): " RPI_USER
read -p "Enter Raspberry Pi IP address: " RPI_IP

# We will use standard SSH which prompts securely for the password natively.
# This avoids installing sshpass and handles passwords securely by default.
echo "[INFO] You will be prompted for the SSH password multiple times during this process."
echo "[INFO] For a seamless experience, consider setting up SSH keys (ssh-copy-id)."

# Verify connectivity
echo ""
echo "[3/6] Verifying SSH connectivity..."
if ! ssh -q -o BatchMode=yes -o ConnectTimeout=5 "${RPI_USER}@${RPI_IP}" exit >/dev/null 2>&1; then
    echo "[INFO] Password required for connection."
    # Just a simple check to see if the host is reachable
    if ! ping -c 1 -W 2 "${RPI_IP}" >/dev/null 2>&1; then
        echo "[ERROR] Host ${RPI_IP} is unreachable."
        exit 1
    fi
fi

# Prepare Raspberry Pi (install dependencies)
echo ""
echo "[4/6] Preparing Raspberry Pi (Dependencies)..."
ssh -t "${RPI_USER}@${RPI_IP}" << 'EOF'
    set -euo pipefail
    echo "Updating system and installing Git..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq git

    if ! command -v docker >/dev/null 2>&1; then
        echo "Installing Docker..."
        curl -fsSL https://get.docker.com | sh
        sudo usermod -aG docker "$USER"
        echo "Docker installed. Note: You may need to reconnect for group changes to take effect."
    else
        echo "Docker is already installed."
    fi
EOF

# Clone or update repository
echo ""
echo "[5/6] Syncing project repository on Raspberry Pi..."
# Using rsync to transfer the whole project including local uncommitted changes,
# which is usually preferred over git clone for local edge deployments.
# However, as requested, we will use git clone/pull, but we need the repo URL.
# Instead of hardcoding the URL, we can infer it or just use rsync.
# Wait, user said: "Clonar el repositorio del proyecto o actualizarlo si ya existe."
# If we clone, we need the origin URL.
REPO_URL=$(git -C "${PROJECT_ROOT}" config --get remote.origin.url || echo "")

if [ -z "${REPO_URL}" ]; then
    echo "[ERROR] Could not determine Git remote URL. Is this a Git repository?"
    exit 1
fi

ssh -t "${RPI_USER}@${RPI_IP}" "
    if [ ! -d \"edge-vision-system\" ]; then
        echo 'Cloning repository...'
        git clone \"${REPO_URL}\" edge-vision-system
    else
        echo 'Updating repository...'
        cd edge-vision-system && git pull
    fi
"

# Transfer models
echo ""
echo "[6/6] Transferring models to Raspberry Pi..."
MODELS_DIR="${PROJECT_ROOT}/services/detector/models"
RPI_MODELS_DIR="edge-vision-system/services/detector/models"

ssh "${RPI_USER}@${RPI_IP}" "mkdir -p ${RPI_MODELS_DIR}"
# Use scp to securely copy the models. -r for directories.
scp -r "${MODELS_DIR}/"* "${RPI_USER}@${RPI_IP}:${RPI_MODELS_DIR}/"

# Execute start_rpi.sh on the Raspberry Pi
echo ""
echo "[Deploy Complete] Starting system on Raspberry Pi..."
ssh -t "${RPI_USER}@${RPI_IP}" "
    cd edge-vision-system
    bash scripts/start_rpi.sh
"
