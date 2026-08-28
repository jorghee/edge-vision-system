# Configuration and Deployment

This document covers installation and execution strategies for the Edge Vision System. The project offers two deployment modes: a fully containerized model for development (laptop/x86_64), and a hybrid model for IoT devices (Raspberry Pi) where hardware constraints require native execution of the detector.

## Prerequisites

| Requirement | Detail |
| :--- | :--- |
| Docker & Docker Compose | Container orchestration for infrastructure services. |
| Python 3.11+ | Runtime for the detector and action service. |
| Git | Repository management and code synchronization. |
| Network | Both devices (laptop and RPi) must be on the same LAN for deployment. |

---

## 1. Laptop Deployment (x86_64)

All components, including the detector, run inside Docker. This approach guarantees environment parity and avoids system dependency conflicts.

### Automated Execution

```bash
bash scripts/start_laptop.sh
```

This script:
1. Builds and starts all Docker services defined in `docker-compose.yml`.
2. Waits for eKuiper to become ready.
3. Provisions eKuiper streams and SQL rules via `scripts/setup_ekuiper.sh`.

> [!NOTE]
> The USB webcam is mapped as `/dev/video2` by default. Adjust `CAMERA_INDEX` and the `devices` section in `docker-compose.yml` to match your hardware.

### Monitoring Events

```bash
docker exec mqtt-broker mosquitto_sub -t "camera/events" -v    # raw detections
docker exec mqtt-broker mosquitto_sub -t "edge/alerts" -v       # filtered alerts
docker exec mqtt-broker mosquitto_sub -t "edge/actions" -v      # action responses
```

### Stopping

```bash
docker compose down
```

---

## 2. Raspberry Pi Deployment (ARM64)

The production deployment targets Raspberry Pi 4 with Raspberry Pi OS Bookworm (64-bit) and a CSI camera (e.g., Camera Rev 1.3). The strategy is hybrid: infrastructure runs in Docker while the detector runs natively.

### Architecture Constraints

| Constraint | Solution |
| :--- | :--- |
| CSI cameras use `libcamera`/`picamera2`, not `/dev/videoX` | Detector runs natively outside Docker. |
| PyTorch inference is slow on ARM (~5s/frame) | Models are exported to NCNN format (NEON SIMD). |
| `picamera2` is a system package on RPi OS | Virtual environment uses `--system-site-packages`. |

### Automated Deployment (from laptop)

The `deploy.sh` script automates the entire process from the laptop:

```bash
bash scripts/deploy.sh
```

It performs the following steps:

| Step | Action | Device |
| :--- | :--- | :--- |
| 1 | Download YOLO models and export to NCNN | Laptop |
| 2 | Push local commits to remote | Laptop |
| 3 | Verify SSH connectivity | Laptop → RPi |
| 4 | Install Git, Docker, picamera2, libcamera | RPi |
| 5 | Clone/pull repository | RPi |
| 6 | Transfer NCNN models | Laptop → RPi |
| 7 | Start infrastructure + detector | RPi |

### Manual Deployment (step by step)

If you prefer manual control, or need to debug individual steps:

#### a. Prepare models on the laptop

```bash
bash scripts/prepare_models.sh
```

This creates a virtual environment, downloads the YOLO base model and the PPE fine-tuned model, and exports both to NCNN format in `services/detector/models/`.

#### b. Start infrastructure on the RPi

```bash
cd ~/edge-vision-system
docker compose -f docker-compose.rpi.yml up --build -d
```

#### c. Provision eKuiper rules

```bash
bash scripts/setup_ekuiper.sh
```

Verify:
```bash
curl -s http://localhost:9081/rules | python3 -m json.tool
curl -s http://localhost:9081/streams | python3 -m json.tool
```

#### d. Prepare the detector environment

```bash
cd services/detector
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements-rpi.txt
```

#### e. Transfer models from laptop

On the laptop:
```bash
scp -r services/detector/models/* pi@<RPI_IP>:~/edge-vision-system/services/detector/models/
```

#### f. Run the detector

```bash
bash scripts/run_rpi.sh
```

The script auto-selects the NCNN model if available and initializes the Picamera2 backend.

---

## Model Conversion (NCNN)

Real-time inference on ARM benefits from the NCNN engine, which exploits NEON SIMD instructions. The conversion should be done on the laptop (faster) and the result transferred to the RPi.

```bash
# On the laptop
cd services/detector/scripts
python3 export_model.py --base ../models/yolov8n.pt --format ncnn
python3 export_model.py --base ../models/ppe_detector.pt --format ncnn
```

Output directories (`yolov8n_ncnn_model/`, `ppe_detector_ncnn_model/`) are generated alongside the source `.pt` files.

> [!TIP]
> The `prepare_models.sh` script automates this entire process, including download and export. It is idempotent and skips steps if models already exist.

---

## Troubleshooting

| Problem | Cause | Solution |
| :--- | :--- | :--- |
| `Could not open Picamera2` | CSI camera not enabled or cable disconnected | Run `sudo raspi-config` → Interface Options → Camera → Enable. Check flat cable. |
| `ModuleNotFoundError: picamera2` | venv created without `--system-site-packages` | Recreate: `python3 -m venv --system-site-packages venv` |
| eKuiper does not receive events | Detector pointing to wrong broker | Verify `MQTT_BROKER=localhost` and port `1883` is exposed in docker-compose. |
| Slow inference (~5+ sec/frame) | Using `.pt` model instead of NCNN | Check `MODEL_PATH` points to `yolov8n_ncnn_model/`. |
| `docker: permission denied` | User not in `docker` group | Run `sudo usermod -aG docker $USER` then re-login, or use `sg docker -c "..."`. |
| `docker compose` fails on RPi | Image not available for ARM64 | Use `docker-compose.rpi.yml` which specifies `slim` instead of `alpine` images. |
