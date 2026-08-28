# Edge Vision System — Documentation

Technical documentation for the Edge Vision System. This project implements a lightweight computer vision solution for Edge Computing, detecting people and verifying the use of Personal Protective Equipment (PPE) — helmets and vests — on resource-constrained IoT devices.

## Overview

The system captures video in real-time from a local camera, processes frames to identify people, verifies PPE compliance, and publishes structured events via MQTT. A lightweight rules engine (eKuiper) filters noise at the edge, forwarding only critical alerts to an action service.

## Table of Contents

| Document | Description |
| :--- | :--- |
| [Architecture and Data Flow](architecture.md) | Component diagram, MQTT communication flow, eKuiper's role. |
| [Detection System](detector.md) | Camera abstraction, YOLO models, PPE evaluation, health monitor. |
| [Configuration and Deployment](deployment.md) | Automated and manual deployment, model conversion, troubleshooting. |

## Project Structure

```
edge-vision-system/
├── services/
│   ├── detector/                 # Vision service
│   │   ├── src/                  # detector.py, camera.py, health_monitor.py
│   │   ├── scripts/              # download_model.py, export_model.py
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── requirements-rpi.txt
│   └── action_service/           # Alert response service
│       ├── src/                  # action_service.py
│       ├── Dockerfile
│       └── requirements.txt
├── infrastructure/
│   └── mqtt/config/              # mosquitto.conf
├── scripts/                      # Deployment automation
│   ├── deploy.sh                 # Full laptop → RPi deployment
│   ├── prepare_models.sh         # Model download and NCNN export
│   ├── start_laptop.sh           # Local execution (full Docker)
│   ├── start_rpi.sh              # RPi execution (hybrid)
│   ├── run_rpi.sh                # Native detector launch
│   └── setup_ekuiper.sh          # SQL rules provisioning
├── docs/                         # Technical documentation
├── docker-compose.yml            # x86_64 orchestration
└── docker-compose.rpi.yml        # ARM64 orchestration (RPi)
```

## Key Dependencies

| Dependency | Purpose |
| :--- | :--- |
| Docker & Docker Compose | Infrastructure orchestration (MQTT, eKuiper, Action Service). |
| Python 3.11+ | Runtime for the detector and action service. |
| Ultralytics (YOLOv8) | Object detection framework. |
| OpenCV / Picamera2 | Image capture (USB webcam / RPi CSI camera). |
| Eclipse Mosquitto | MQTT broker. |
| LF Edge eKuiper | Edge stream processing and SQL rules engine. |
| NCNN | Optimized neural network inference on ARM (NEON SIMD). |
