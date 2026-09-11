"""
eKuiper Portable Python Plugin: PPE Inference.

Contains two components registered with eKuiper:

1. **cameraSource**: A Portable Source that captures frames from a local
   RTSP stream (served by the MediaMTX sidecar container, which reads the
   RPi CSI camera via libcamera). Frames are base64-encoded JPEG strings.

2. **ppeInference**: A Portable Function that receives a base64 frame,
   decodes it, runs YOLOv8 person detection, analyzes PPE compliance
   (helmet + vest), and returns structured detection results.

IMPORTANT: All heavy imports (cv2, numpy, ultralytics) are deferred
to first use. The module-level code must be fast (<1s) so that the
eKuiper IPC handshake completes before the timeout.
"""

import os
import time
import base64
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PPE_PLUGIN] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

try:
    from ekuiper import Function, Context, Source
except ImportError:
    class Function:
        def validate(self, args): return ""
        def exec(self, args, ctx): return None
        def is_aggregate(self): return False

    class Source:
        def configure(self, datasource, conf): pass
        def open(self, ctx): pass
        def close(self, ctx): pass

    class Context:
        pass

MODELS_DIR = os.getenv("MODELS_DIR", "/kuiper/models")
CONFIDENCE_THR = float(os.getenv("CONFIDENCE_THR", "0.45"))
CAMERA_ID = os.getenv("CAMERA_ID", "cam-rpi-01")
# RTSP URL served by MediaMTX sidecar container
RTSP_URL = os.getenv("RTSP_URL", "rtsp://mediamtx:8554/cam")
CAMERA_FPS = float(os.getenv("CAMERA_FPS", "0.5"))

# Lazy-loaded references
_models = None
_cv2 = None
_np = None

HELMET_CLASS = 0
HEAD_CLASS = 1


def _get_cv2():
    """Lazy import of OpenCV."""
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
        log.info("OpenCV loaded: %s", cv2.__version__)
    return _cv2


def _get_np():
    """Lazy import of numpy."""
    global _np
    if _np is None:
        import numpy
        _np = numpy
    return _np


# Portable Source: Camera Capture via RTSP
class CameraSource(Source):
    """Captures frames from RTSP stream (MediaMTX sidecar)."""

    def configure(self, datasource: str, conf: dict):
        self.rtsp_url = conf.get("url", RTSP_URL)
        self.interval = 1.0 / float(conf.get("fps", CAMERA_FPS))
        self.cap = None
        log.info("CameraSource configured: url=%s, fps=%s",
                 self.rtsp_url, conf.get("fps", CAMERA_FPS))

    def open(self, ctx: Context):
        cv2 = _get_cv2()
        np = _get_np()

        # Retry connection to RTSP (MediaMTX may still be starting)
        for attempt in range(30):
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if self.cap.isOpened():
                break
            log.warning("RTSP not ready (attempt %d/30), retrying...",
                        attempt + 1)
            time.sleep(2)

        if not self.cap or not self.cap.isOpened():
            log.error("Failed to connect to RTSP stream: %s", self.rtsp_url)
            return

        log.info("Connected to RTSP stream: %s", self.rtsp_url)

        # Model warm-up
        # Loading a YOLO model into RAM is not enough.  The first real
        # inference triggers NCNN/TFLite graph compilation and hardware
        # buffer allocation, which takes ~10 s on ARM64 — well beyond
        # eKuiper's 5 s IPC receive timeout.  Running a dummy inference
        # on a synthetic black frame forces that one-time cost here,
        # during source init, where there is no timeout pressure.
        try:
            models = _load_models()
            log.info("AI models loaded, running warm-up inference...")
            dummy = np.zeros((320, 240, 3), dtype=np.uint8)
            models["base"](dummy, conf=0.9, verbose=False)
            if models["ppe"] is not None:
                models["ppe"](dummy, conf=0.9, verbose=False)
            log.info("Warm-up inference complete — models are hot")
        except Exception as e:
            log.warning(
                "Model warm-up failed (will retry on first call): %s", e)

        # Active-drain capture loop
        # An RTSP camera pushes ~30 fps continuously.  The old pattern
        # of read-one-frame then sleep(interval) let ~60 frames pile up
        # in FFmpeg's internal H264 decode buffer during each sleep,
        # eventually corrupting packets (bytestream -5/-7/-9).
        #
        # Instead, we read every frame at full speed (keeping the
        # buffer empty) but only encode + emit to eKuiper when enough
        # wall-clock time has elapsed.  Intermediate frames are simply
        # discarded.
        last_emit = 0.0

        while True:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    log.warning("Failed to read frame, reconnecting...")
                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.rtsp_url)
                    last_emit = 0.0
                    continue

                now = time.time()
                if now - last_emit < self.interval:
                    # Drain the buffer — discard this frame silently
                    continue

                # Encode frame as JPEG bytes, then base64 for transport
                _, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                b64_frame = base64.b64encode(buf.tobytes()).decode("ascii")

                ctx.emit({"frame": b64_frame, "camera_id": CAMERA_ID,
                          "timestamp": datetime.utcnow().isoformat() + "Z"},
                         {})

                last_emit = now
            except Exception as e:
                log.error("Error in CameraSource loop: %s", e)
                time.sleep(2)

    def close(self, ctx: Context):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            log.info("RTSP stream released")


# AI Model Loading (lazy)
def _load_models():
    global _models
    if _models is not None:
        return _models

    from ultralytics import YOLO
    import ultralytics
    ultralytics.checks = lambda: None
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    base_candidates = [
        os.path.join(MODELS_DIR, "yolov8n_ncnn_model"),
        os.path.join(MODELS_DIR, "yolov8n_saved_model",
                     "yolov8n_float32.tflite"),
        os.path.join(MODELS_DIR, "yolov8n.tflite"),
        os.path.join(MODELS_DIR, "yolov8n.onnx"),
        os.path.join(MODELS_DIR, "yolov8n.pt"),
    ]
    ppe_candidates = [
        os.path.join(MODELS_DIR, "ppe_detector_ncnn_model"),
        os.path.join(MODELS_DIR, "ppe_detector_saved_model",
                     "ppe_detector_float32.tflite"),
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
        log.warning("PPE model not found, using HSV color fallback")

    _models = {"base": base_model, "ppe": ppe_model}
    log.info("Models loaded successfully")
    return _models


# PPE Detection Helpers
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

    if not head_detected and not helmet_detected:
        # Cannot assess helmet status: head region not visible or not
        # classifiable. Return None to signal an indeterminate result
        # instead of fabricating a confidence value.
        return None, 0.0

    return helmet_detected, round(helmet_conf, 2)


def _check_helmet_hsv(crop):
    """Fallback: detect helmet by HSV color analysis in the head region."""
    cv2 = _get_cv2()
    np = _get_np()

    h = crop.shape[0]
    head = crop[0:int(h * 0.30), :]
    if head.size == 0:
        return False, 0.0

    hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
    total = head.shape[0] * head.shape[1]

    color_ranges = [
        ([15, 80, 80], [35, 255, 255]),    # yellow
        ([5, 80, 80], [15, 255, 255]),     # orange
        ([0, 0, 180], [180, 30, 255]),     # white
        ([0, 100, 100], [5, 255, 255]),    # red lower
        ([170, 100, 100], [180, 255, 255]) # red upper
    ]

    for lo, hi in color_ranges:
        mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        ratio = cv2.countNonZero(mask) / total
        if ratio > 0.08:
            return True, round(min(0.99, ratio * 8), 2)

    return False, 0.0


def _check_vest(crop):
    """Detect reflective vest by HSV color in the torso region."""
    cv2 = _get_cv2()
    np = _get_np()

    h = crop.shape[0]
    torso = crop[int(h * 0.30):int(h * 0.70), :]
    if torso.size == 0:
        return None, 0.0

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    total = torso.shape[0] * torso.shape[1]

    ranges = [
        ([5, 150, 150], [20, 255, 255]),
        ([20, 150, 150], [40, 255, 255]),
    ]
    ratio = sum(
        cv2.countNonZero(cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8)))
        / total for lo, hi in ranges
    )
    return ratio > 0.12, round(min(0.99, ratio * 6), 2)


def _classify_severity(helmet_ok, vest_ok):
    """Determine event type and severity from PPE status.

    Uses a three-valued logic: True (detected), False (not detected),
    and None (indeterminate -- the model could not assess the item).
    When any assessment is indeterminate, the entire event is classified
    as indeterminate to avoid false positives from fabricated values.
    """
    if helmet_ok is None or vest_ok is None:
        return "indeterminate", "indeterminate"
    if not helmet_ok and not vest_ok:
        return "no_helmet_no_vest", "critical"
    elif not helmet_ok:
        return "no_helmet", "high"
    elif not vest_ok:
        return "no_vest", "high"
    return "ppe_compliant", "none"


def process_frame(frame_bytes):
    """Core inference pipeline: frame bytes -> list of detection dicts."""
    cv2 = _get_cv2()
    np = _get_np()
    models = _load_models()

    frame = cv2.imdecode(
        np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR
    )
    if frame is None:
        return {"event_type": "error", "severity": "none",
                "confidence": 0.0}

    persons = _detect_persons(frame, models)

    if not persons:
        return {
            "camera_id": CAMERA_ID,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "clear",
            "severity": "none",
            "confidence": 0.99,
            "persons_detected": 0,
        }

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

        valid_confs = [person_conf]
        if helmet_ok is not None and helmet_ok:
            valid_confs.append(helmet_conf)
        if vest_ok is not None and vest_ok:
            valid_confs.append(vest_conf)

        avg_conf = round(sum(valid_confs) / len(valid_confs), 2)

        detections.append({
            "event_type": event_type,
            "severity": severity,
            "confidence": avg_conf,
            "person_id": person_id,
            "helmet_detected": bool(helmet_ok) if helmet_ok is not None else False,
            "helmet_confidence": round(helmet_conf, 2),
            "vest_detected": vest_ok,
            "vest_confidence": round(vest_conf, 2)
        })

    if not detections:
        return {
            "camera_id": CAMERA_ID,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "highest_severity": "none",
            "event_type": "clear",
            "confidence": 0.99,
            "persons_detected": len(persons),
            "violators": []
        }

    # Sort by severity: critical > high > indeterminate > none
    severity_rank = {"critical": 0, "high": 1, "indeterminate": 2, "none": 3}
    detections.sort(key=lambda d: severity_rank.get(d["severity"], 4))

    highest = detections[0]

    result = {
        "camera_id": CAMERA_ID,
        "timestamp": highest["timestamp"],
        "highest_severity": highest["severity"],
        "event_type": highest["event_type"],
        "confidence": highest["confidence"],
        "persons_detected": len(persons),
        "violators": detections
    }

    # Attach a compressed thumbnail for critical/high alerts (for Grafana)
    if highest["severity"] in ("critical", "high"):
        try:
            thumb = cv2.resize(frame, (320, 240))
            _, buf = cv2.imencode(
                ".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 50])
            result["snapshot"] = base64.b64encode(
                buf.tobytes()).decode("ascii")
        except Exception as e:
            log.warning("Failed to generate snapshot: %s", e)

    return result


# Portable Function: PPE Inference
class PpeInference(Function):
    """eKuiper Portable Plugin function.

    Called from SQL rules as: ppeInference(frame)
    Receives base64-encoded frame string, returns list of detection results.
    """

    def __init__(self):
        pass

    def validate(self, args: list) -> str:
        if len(args) != 1:
            return "ppeInference requires exactly one argument (frame data)"
        return ""

    def exec(self, args: list, ctx: Context) -> dict:
        try:
            frame_data = args[0]

            if isinstance(frame_data, str):
                frame_bytes = base64.b64decode(frame_data)
            else:
                frame_bytes = frame_data

            return process_frame(frame_bytes)

        except Exception as e:
            log.error("Inference error: %s", e, exc_info=True)
            return {"event_type": "error", "severity": "none",
                    "error": str(e)}

    def is_aggregate(self) -> bool:
        return False


# Plugin entry point
if __name__ == '__main__':
    import sys
    if "--info" in sys.argv:
        print("\n" + "="*60)
        print("  eKuiper PPE Inference Module - Traceability Info")
        print("="*60)
        try:
            models = _load_models()
            base_model = models["base"]
            ppe_model = models["ppe"]
            
            print(f"[Model] Base YOLO (Persons): {base_model.model_name if hasattr(base_model, 'model_name') else 'Loaded from ' + MODELS_DIR}")
            if ppe_model:
                print(f"[Model] PPE (Helmet): {ppe_model.model_name if hasattr(ppe_model, 'model_name') else 'Loaded from ' + MODELS_DIR}")
            else:
                print("[Model] PPE (Helmet): NONE -> Using auxiliary HSV Logic")
            
            print(f"[Config] Models Directory: {MODELS_DIR}")
            print(f"[Config] Confidence Threshold: {CONFIDENCE_THR}")
            print(f"[Config] Camera FPS Limit: {CAMERA_FPS}")
            print(f"[Config] Camera ID: {CAMERA_ID}")
            print("="*60 + "\n")
            sys.exit(0)
        except Exception as e:
            print(f"Error gathering info: {e}")
            sys.exit(1)

    from ekuiper.runtime.plugin import PluginConfig, start
    c = PluginConfig(
        name="ppe_inference",
        sources={"cameraSource": lambda: CameraSource()},
        sinks={},
        functions={"ppeInference": lambda: PpeInference()},
    )
    start(c)
