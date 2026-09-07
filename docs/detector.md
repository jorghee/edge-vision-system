# Detection System

The vision intelligence resides in the eKuiper Portable Python Plugin (`services/ekuiper/plugins/ppe_inference/ppe_func.py`). This plugin replaces the standalone `detector.py` service and runs inside eKuiper's pipeline.

## Pipeline Overview

```
Video Source → ppeInference(frame) → SQL Rules → MQTT Sink
```

The plugin receives binary frame data from eKuiper's Video Source, performs all detection and classification, and returns structured JSON results that eKuiper's SQL engine can filter.

## Person Detection (YOLOv8)

Each frame is processed with YOLOv8 nano (`yolov8n`), filtering exclusively for COCO class 0 (Person). The model is loaded in TFLite format for eKuiper compatibility, with automatic fallback to ONNX or PT.

| Format | Priority | Use Case |
|:---|:---|:---|
| TFLite | Primary | eKuiper native AI functions |
| ONNX | Secondary | Universal compatibility |
| PT (PyTorch) | Fallback | Development/testing |

Model search paths are configured via the `MODELS_DIR` environment variable (default: `/kuiper/models`).

## PPE Detection (Helmets and Vests)

Once people are located (bounding boxes), the plugin crops each person region and applies two concurrent strategies:

### Primary: Fine-Tuned Model (Helmet)

If `ppe_detector.tflite` (or `.onnx`/`.pt`) is available, the plugin runs inference on the person crop. This model (`keremberke/yolov8n-hard-hat-detection`) detects:

| Class ID | Label |
|:---|:---|
| 0 | Helmet detected |
| 1 | Head without helmet |

When neither head nor helmet is detected (person facing away), the plugin assumes compliance to avoid false positives.

### Fallback: HSV Color Analysis

When the PPE model is unavailable, or for vest detection (which the fine-tuned model does not cover), HSV color segmentation is applied:

| Step | Detail |
|:---|:---|
| Region splitting | Upper 30% (head) for helmet, middle 30-70% (torso) for vest |
| Color space | BGR → HSV conversion |
| Mask evaluation | Pixel density against yellow, orange, white, red masks |
| Threshold | Helmet: >8% density, Vest: >12% density |

## Severity Classification

| Severity | Event Type | Condition |
|:---|:---|:---|
| `critical` | `no_helmet_no_vest` | Both items missing |
| `high` | `no_helmet` / `no_vest` | Partial PPE absence |
| `none` | `ppe_compliant` | Both detected |
| `none` | `clear` | No person in frame |

## Health Monitor

The module `services/detector/src/health_monitor.py` runs as a companion process publishing to `edge/health` every 30 seconds:

| Metric | Source |
|:---|:---|
| CPU temperature (°C) | `/sys/class/thermal/` |
| CPU usage (%) | Load average |
| RAM (total, used, %) | `/proc/meminfo` |
| Disk usage | Root partition |
| SoC throttling state | `vcgencmd get_throttled` |

## Plugin Architecture

The Portable Plugin communicates with eKuiper via nanomsg (IPC). It runs as an independent Python process managed by eKuiper's plugin lifecycle:

| File | Purpose |
|:---|:---|
| `ppe_func.py` | Inference logic + `PpeInference` class implementing `Function` interface |
| `ppe_inference.json` | Plugin metadata (entry point, function names) |
| `requirements.txt` | Python dependencies (ultralytics, opencv, numpy) |

The plugin lazy-loads models on the first inference call to minimize startup time.
