# Edge Vision System — Documentation

Technical documentation for the Edge Vision System. This project implements an eKuiper-native computer vision solution for Edge Computing, detecting people and verifying Personal Protective Equipment (PPE) compliance on resource-constrained IoT devices.

## Overview

eKuiper captures video frames directly from the local camera, processes them through a Portable Python Plugin (YOLOv8 + PPE analysis), and publishes only filtered alerts via MQTT. No standalone detector process is needed.

## Table of Contents

| Document | Description |
| :--- | :--- |
| [Architecture and Data Flow](architecture.md) | eKuiper-native pipeline, MQTT topics, deployment model. |
| [Detection System](detector.md) | Portable Plugin, YOLO models, PPE evaluation, health monitor. |
| [Configuration and Deployment](deployment.md) | Automated/manual deployment, model conversion, troubleshooting. |

## Key Dependencies

| Dependency | Purpose |
| :--- | :--- |
| Docker & Docker Compose | All services run in containers. |
| LF Edge eKuiper | Video capture, AI inference pipeline, SQL rules engine. |
| Ultralytics (YOLOv8) | Object detection (runs inside Portable Plugin). |
| OpenCV | Image processing for HSV color analysis. |
| Eclipse Mosquitto | MQTT broker (sink for filtered alerts). |
| Python 3.11+ | Portable Plugin runtime + model export scripts. |
