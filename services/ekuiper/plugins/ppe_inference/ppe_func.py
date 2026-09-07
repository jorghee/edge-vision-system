"""
eKuiper Portable Python Plugin: PPE Inference.

Receives a binary video frame from eKuiper's Video Source, runs
YOLOv8 person detection, analyzes PPE compliance (helmet + vest),
and returns structured detection results.

This replaces the standalone detector.py service. All inference
logic runs inside eKuiper's pipeline via the Portable Plugin SDK.
"""

import os
import logging
from datetime import datetime

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PPE_PLUGIN] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

try:
    from ekuiper import Function, Context
except ImportError:
    # Allow local testing without eKuiper SDK
    class Function:
        def validate(self, args): return ""
        def exec(self, args, ctx): return None
        def is_aggregate(self): return False
    class Context:
        pass

MODELS_DIR = os.getenv("MODELS_DIR", "/kuiper/models")
CONFIDENCE_THR = float(os.getenv("CONFIDENCE_THR", "0.45"))
CAMERA_ID = os.getenv("CAMERA_ID", "cam-rpi-01")

# Models are lazy-loaded on first inference call
_models = None


def _load_models():
    global _models
    if _models is not None:
        return _models

    from ultralytics import YOLO
    import ultralytics
    ultralytics.checks = lambda: None
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    # Try TFLite first, then ONNX, then .pt as fallback
    base_candidates = [
        os.path.join(MODELS_DIR, "yolov8n_saved_model", "yolov8n_float32.tflite"),
        os.path.join(MODELS_DIR, "yolov8n.tflite"),
        os.path.join(MODELS_DIR, "yolov8n.onnx"),
        os.path.join(MODELS_DIR, "yolov8n.pt"),
    ]
    ppe_candidates = [
        os.path.join(MODELS_DIR, "ppe_detector_saved_model", "ppe_detector_float32.tflite"),
        os.path.join(MODELS_DIR, "ppe_detector.tflite"),
        os.path.join(MODELS_DIR, "ppe_detector.onnx"),
        os.path.join(MODELS_DIR, "ppe_detector.pt"),
    ]

    base_path = next((p for p in base_candidates if os.path.exists(p)), None)
    ppe_path = next((p for p in ppe_candidates if os.path.exists(p)), None)

    if base_path is None:
        log.error("No base YOLO model found in %s", MODELS_DIR)
        raise FileNotFoundError(f"No YOLO model in {MODELS_DIR}")

    log.info("Loading base model: %s", base_path)
    base_model = YOLO(base_path)

    ppe_model = None
    if ppe_path:
        log.info("Loading PPE model: %s", ppe_path)
        ppe_model = YOLO(ppe_path)
    else:
        log.warning("PPE model not found, using HSV color fallback for helmets")

    _models = {"base": base_model, "ppe": ppe_model}
    log.info("Models loaded successfully")
    return _models


# PPE detection class indices (from keremberke/yolov8n-hard-hat-detection)
HELMET_CLASS = 0
HEAD_CLASS = 1


def _detect_persons(frame, models):
    """Run YOLOv8 on full frame, return person bounding boxes."""
    results = models["base"](
        frame, conf=CONFIDENCE_THR, classes=[0], verbose=False
    )
    persons = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            persons.append((x1, y1, x2, y2, conf))
    return persons


def _check_helmet_model(crop, models):
    """Check helmet using fine-tuned PPE model."""
    results = models["ppe"](crop, conf=CONFIDENCE_THR, verbose=False)
    helmet_detected = False
    helmet_conf = 0.0
    head_detected = False

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == HELMET_CLASS:
                helmet_detected = True
                helmet_conf = max(helmet_conf, conf)
            elif cls == HEAD_CLASS:
                head_detected = True

    # If neither head nor helmet visible, assume compliant (person facing away)
    if not head_detected and not helmet_detected:
        return True, 0.5

    return helmet_detected, round(helmet_conf, 2)


def _check_helmet_hsv(crop):
    """Fallback: detect helmet by HSV color analysis in the head region."""
    h = crop.shape[0]
    head = crop[0:int(h * 0.30), :]
    if head.size == 0:
        return False, 0.0

    hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
    total = head.shape[0] * head.shape[1]

    color_ranges = [
        ([15, 80, 80], [35, 255, 255]),   # yellow
        ([5, 80, 80], [15, 255, 255]),     # orange
        ([0, 0, 180], [180, 30, 255]),     # white
        ([0, 100, 100], [5, 255, 255]),    # red
    ]

    for lo, hi in color_ranges:
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        ratio = cv2.countNonZero(mask) / total
        if ratio > 0.08:
            return True, round(min(0.99, ratio * 8), 2)

    return False, 0.0


def _check_vest(crop):
    """Detect reflective vest by HSV color in the torso region."""
    h = crop.shape[0]
    torso = crop[int(h * 0.30):int(h * 0.70), :]
    if torso.size == 0:
        return False, 0.0

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    total = torso.shape[0] * torso.shape[1]

    ranges = [
        ([5, 150, 150], [20, 255, 255]),
        ([20, 150, 150], [40, 255, 255]),
    ]
    ratio = sum(
        cv2.countNonZero(cv2.inRange(hsv, np.array(lo), np.array(hi))) / total
        for lo, hi in ranges
    )
    return ratio > 0.12, round(min(0.99, ratio * 6), 2)


def _classify_severity(helmet_ok, vest_ok):
    """Determine event type and severity from PPE status."""
    if not helmet_ok and not vest_ok:
        return "no_helmet_no_vest", "critical"
    elif not helmet_ok:
        return "no_helmet", "high"
    elif not vest_ok:
        return "no_vest", "high"
    return "ppe_compliant", "none"


def process_frame(frame_bytes):
    """Core inference pipeline: frame bytes → list of detection dicts."""
    models = _load_models()

    frame = cv2.imdecode(
        np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR
    )
    if frame is None:
        return [{"event_type": "error", "severity": "none", "confidence": 0.0}]

    persons = _detect_persons(frame, models)

    if not persons:
        return [{
            "camera_id": CAMERA_ID,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "clear",
            "severity": "none",
            "confidence": 0.99,
            "persons_detected": 0,
        }]

    detections = []
    for person_id, (x1, y1, x2, y2, person_conf) in enumerate(persons):
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        if models["ppe"] is not None:
            helmet_ok, helmet_conf = _check_helmet_model(crop, models)
        else:
            helmet_ok, helmet_conf = _check_helmet_hsv(crop)

        vest_ok, vest_conf = _check_vest(crop)
        event_type, severity = _classify_severity(helmet_ok, vest_ok)

        avg_conf = round((person_conf + helmet_conf + vest_conf) / 3, 2)

        detections.append({
            "camera_id": CAMERA_ID,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "severity": severity,
            "confidence": avg_conf,
            "person_id": person_id,
            "helmet_detected": helmet_ok,
            "helmet_confidence": round(helmet_conf, 2),
            "vest_detected": vest_ok,
            "vest_confidence": round(vest_conf, 2),
            "persons_detected": len(persons),
        })

    return detections


class PpeInference(Function):
    """eKuiper Portable Plugin function.

    Called from SQL rules as: ppeInference(self)
    Receives binary frame data, returns list of detection results.
    """

    def validate(self, args: list) -> str:
        if len(args) != 1:
            return "ppeInference requires exactly one argument (binary frame)"
        return ""

    def exec(self, args: list, ctx: Context) -> list:
        try:
            frame_bytes = args[0]
            if isinstance(frame_bytes, str):
                import base64
                frame_bytes = base64.b64decode(frame_bytes)
            return process_frame(frame_bytes)
        except Exception as e:
            log.error("Inference error: %s", e, exc_info=True)
            return [{"event_type": "error", "severity": "none", "error": str(e)}]

    def is_aggregate(self) -> bool:
        return False


# Plugin entry point for eKuiper
ppeInference = PpeInference()
