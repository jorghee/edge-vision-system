# Detection System

The core of the computer vision pipeline resides in the detector module (`services/detector/src/`). Its primary responsibility is processing raw images, identifying people in the scene, and evaluating Personal Protective Equipment (PPE) compliance.

## Camera Abstraction

The module `services/detector/src/camera.py` provides a unified interface (`Camera` protocol) supported by two backends:

| Backend | Class | Use Case | Library |
| :--- | :--- | :--- | :--- |
| **OpenCV** | `OpenCVCamera` | USB webcams on x86_64 (laptops/servers) | `cv2.VideoCapture` |
| **Picamera2** | `PiCamera` | CSI cameras on Raspberry Pi | `picamera2` + `libcamera` |

Backend selection is controlled by the `CAMERA_BACKEND` environment variable (`"opencv"`, `"picamera2"`, or `"auto"`). In `auto` mode, the factory attempts Picamera2 first and falls back to OpenCV.

## Base Detection (People)

Each frame is processed with **YOLOv8** (typically the `yolov8n` nano variant). The network is pre-trained on COCO, providing high-accuracy detection. The detector filters exclusively for class `0` (Person), ignoring all other classes.

On Raspberry Pi, models are exported to **NCNN** format for optimized ARM inference using NEON SIMD instructions. The export is handled by `services/detector/scripts/export_model.py`.

## PPE Detection (Helmets and Vests)

Once people are located (bounding boxes), the detector crops each region and applies two concurrent strategies:

### Primary: Fine-Tuned Model

If a specialized model (`ppe_detector.pt` or its NCNN equivalent) is available, the system runs inference directly on the person crop. This model (`keremberke/yolov8n-hard-hat-detection`) is trained specifically for hard hat detection and provides robust results across varying lighting conditions.

### Fallback: HSV Color Analysis

When the PPE model is unavailable, or for vest detection (which the fine-tuned model does not cover), a classical color segmentation approach is applied:

| Step | Detail |
| :--- | :--- |
| Region splitting | The person crop is divided geometrically: upper third (head), middle section (torso). |
| Color space | The region is converted from BGR to HSV (Hue, Saturation, Value). |
| Mask evaluation | Pixel density is measured against masks for typical PPE colors (yellow, orange, red, white). |
| Threshold | If the target color density exceeds a preset threshold in the area of interest, the item is marked as "detected". |

## Severity Evaluation

Based on helmet and vest presence, the system assigns a classification:

| Severity | Event Type | Condition |
| :--- | :--- | :--- |
| `none` | `ppe_compliant` | Both helmet and vest detected. |
| `high` | `no_helmet` / `no_vest` | Partial PPE absence. |
| `critical` | `no_helmet_no_vest` | Both items missing. |
| `info` | `clear` | No person detected in the frame. |

## Health Monitor

The module `services/detector/src/health_monitor.py` is a companion process that extracts hardware telemetry. It publishes to `edge/health` every 30 seconds:

| Metric | Source |
| :--- | :--- |
| CPU temperature (°C) | `/sys/class/thermal/` |
| CPU usage (%) | Load average |
| RAM (total, used, %) | `/proc/meminfo` |
| Disk usage | Root partition |
| SoC throttling state | `vcgencmd get_throttled` |

This telemetry can be consumed by eKuiper rules to prevent thermal damage and manage device availability autonomously.
