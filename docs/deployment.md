# Configuration and Deployment

All services run in Docker. eKuiper captures video directly from the camera device and performs AI inference internally via a Portable Python Plugin.

## Prerequisites

| Requirement | Laptop | Raspberry Pi |
| :--- | :--- | :--- |
| Docker & Docker Compose | Required | Required |
| Git | Required | Required |
| Python 3.11+ | For model export only | Not needed |
| USB/CSI Camera | `/dev/video2` (default) | `/dev/video0` (default) |

---

## 1. Laptop Deployment (x86_64)

### Automated Execution

```bash
bash scripts/start_laptop.sh
```

This script builds and starts all Docker services, waits for eKuiper, and provisions the video stream and SQL rules.

### Monitoring

```bash
docker exec mqtt-broker mosquitto_sub -t "edge/alerts" -v    # filtered alerts
docker exec mqtt-broker mosquitto_sub -t "edge/monitor" -v   # all non-clear events
```

### Stopping

```bash
docker compose down
```

> [!NOTE]
> The USB webcam is mapped as `/dev/video2` by default. Adjust `devices` in `docker-compose.yml` and `url` in `infrastructure/ekuiper/sources/video.yaml` to match your hardware.

---

## 2. Raspberry Pi Deployment (ARM64)

### Automated Deployment (from laptop)

```bash
bash scripts/deploy.sh
```

| Step | Action | Device |
| :--- | :--- | :--- |
| 1 | Download YOLO models, export to TFLite + NCNN | Laptop |
| 2 | Push local commits to remote | Laptop |
| 3 | Verify SSH connectivity | Laptop → RPi |
| 4 | Install Git and Docker | RPi |
| 5 | Clone/pull repository | RPi |
| 6 | Transfer models | Laptop → RPi |
| 7 | Start all services (docker compose) | RPi |

### Manual Deployment

#### a. Prepare models on the laptop

```bash
bash scripts/prepare_models.sh
```

#### b. Transfer models to RPi

```bash
scp -r services/detector/models/* pi@<RPI_IP>:~/edge-vision-system/services/detector/models/
```

#### c. Start the system on RPi

```bash
cd ~/edge-vision-system
bash scripts/start_rpi.sh
```

---

## Model Conversion

The Portable Plugin supports TFLite, ONNX, and PT formats. TFLite is the primary format for eKuiper compatibility.

```bash
cd services/detector/scripts
python3 export_model.py --base ../models/yolov8n.pt --format tflite
python3 export_model.py --base ../models/ppe_detector.pt --format tflite
```

> [!TIP]
> `prepare_models.sh` automates download + TFLite + NCNN export. It is idempotent.

---

## eKuiper Configuration

### Video Source

Configuration file: `infrastructure/ekuiper/sources/video.yaml`

```yaml
default:
  url: /dev/video0       # Camera device path
  interval: 3000         # Milliseconds between captures
  codec: mjpeg
```

### Rules Provisioning

```bash
bash scripts/setup_ekuiper.sh
```

Verify:
```bash
curl -s http://localhost:9081/rules | python3 -m json.tool
curl -s http://localhost:9081/streams | python3 -m json.tool
```

---

## Troubleshooting

| Problem | Cause | Solution |
| :--- | :--- | :--- |
| `No base YOLO model found` | Models not transferred to RPi | Run `deploy.sh` or `scp` models manually |
| eKuiper container won't start | Device `/dev/video0` not found | Check camera connection, update `video.yaml` |
| No alerts appearing | Plugin not loaded | Check `docker logs ekuiper-engine` for Python errors |
| `docker: permission denied` | User not in docker group | `sudo usermod -aG docker $USER` then re-login |
| Low inference accuracy | INT8 quantization in TFLite | Try `--format onnx` for FP32 precision |
| CSI camera not detected in Docker | libcamera not V4L2-compatible | Install `rpicam-v4l2` shim or use `privileged: true` |
