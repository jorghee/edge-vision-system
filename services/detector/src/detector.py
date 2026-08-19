import ultralytics
import json
import time
import os
import logging
from datetime import datetime

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO

from camera import create_camera, CAMERA_BACKEND

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DETECTOR] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)

ultralytics.checks = lambda: None
logging.getLogger("ultralytics").setLevel(logging.WARNING)

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "camera/events")
CAMERA_ID = os.getenv("CAMERA_ID", "cam-01")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "2"))
INTERVAL_SEC = float(os.getenv("INTERVAL_SEC", "3"))
USE_SIMULATION = os.getenv("USE_SIMULATION", "false").lower() == "true"
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/yolov8n.pt")
CONFIDENCE_THR = float(os.getenv("CONFIDENCE_THR", "0.45"))
CAPTURE_WIDTH = int(os.getenv("CAPTURE_WIDTH", "640"))
CAPTURE_HEIGHT = int(os.getenv("CAPTURE_HEIGHT", "480"))


class EPPDetector:
    """
    Personal Protective Equipment Detector using YOLOv8.

    The yolov8n.pt model was trained on COCO and recognizes 80 classes.
    Class 0 is 'person'. We use this detection as the first filter.

    For PPE we use region analysis on each detected person:
    - Upper region (30% height): head, detect helmet
    - Middle region (40% height): torso, detect vest
    """

    # Relevant COCO classes
    COCO_PERSON_CLASS = 0

    EPP_CLASSES = {
        "helmet":     0,   # helmet detected
        "head":       1,   # head without helmet
        "person":     2,   # person (redundant with base model)
    }

    def __init__(self, model_path: str, confidence: float = 0.45):
        ppe_path = os.path.join(os.path.dirname(model_path), "ppe_detector.pt")

        log.info(f"[YOLO] Loading base model from {model_path}...")
        self.model_base = YOLO(model_path)

        # Load PPE model if it exists
        if os.path.exists(ppe_path):
            log.info(f"[YOLO] Loading PPE model from {ppe_path}...")
            self.model_ppe = YOLO(ppe_path)
            self.use_ppe_model = True
            log.info("[YOLO] PPE model loaded — real helmet inference")
        else:
            log.warning("[YOLO] PPE model not found, using color analysis")
            self.model_ppe = None
            self.use_ppe_model = False

        self.confidence = confidence
        log.info("[YOLO] Models ready")

    def detect_persons(self, frame: np.ndarray) -> list:
        """
        Executes YOLOv8 on the full frame.
        Returns list of bounding boxes for detected persons.
        Each bbox is (x1, y1, x2, y2, confidence).
        """
        results = self.model_base(
            frame,
            conf=self.confidence,
            classes=[self.COCO_PERSON_CLASS],  # Only detect persons
            verbose=False
        )

        persons = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                persons.append((x1, y1, x2, y2, conf))

        return persons

    def analyze_ppe(self, frame: np.ndarray, bbox: tuple) -> dict:
        """
        Analyzes a person's PPE.
        Uses PPE model if available, HSV color as fallback.
        """
        x1, y1, x2, y2, person_conf = bbox
        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            return self._empty_ppe_result()

        if self.use_ppe_model:
            return self._analyze_with_model(person_crop, person_conf, bbox)
        else:
            return self._analyze_with_color(person_crop, person_conf, bbox)

    def _analyze_with_model(self, crop: np.ndarray, person_conf: float, bbox: tuple) -> dict:
        """
        Real inference with fine-tuned model.
        Runs locally on CPU — no internet, no cloud.
        """
        x1, y1, x2, y2, _ = bbox
        results = self.model_ppe(crop, conf=self.confidence, verbose=False)

        helmet_detected = False
        helmet_confidence = 0.0
        head_detected = False   # head without helmet

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == self.EPP_CLASSES["helmet"]:
                    helmet_detected = True
                    helmet_confidence = max(helmet_confidence, conf)

                elif cls == self.EPP_CLASSES["head"]:
                    head_detected = True   # confirms there is a visible head

        # If head detected but no helmet → no_helmet confirmed
        # If neither head nor helmet detected → person facing away, ignore
        if not head_detected and not helmet_detected:
            # Cannot determine — assume compliant to avoid false positives
            helmet_detected = True
            helmet_confidence = 0.5

        return {
            "person_confidence": round(person_conf, 2),
            "helmet": {
                "detected":   helmet_detected,
                "confidence": round(helmet_confidence, 2),
                "color":      None,
                "method":     "yolo_model"
            },
            # This model does not detect vest — use color as complement
            "vest": self._detect_vest_color_from_crop(crop),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        }

    def _analyze_with_color(self, crop: np.ndarray, person_conf: float, bbox: tuple) -> dict:
        """Fallback: HSV color analysis when there is no PPE model."""
        x1, y1, x2, y2, _ = bbox
        height = crop.shape[0]

        head_region = crop[0:int(height * 0.30), :]
        torso_region = crop[int(height * 0.30):int(height * 0.70), :]

        return {
            "person_confidence": round(person_conf, 2),
            "helmet": {**self._detect_helmet_color(head_region), "method": "color_hsv"},
            "vest":   self._detect_vest_color_from_crop(torso_region),
            "bbox":   {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        }

    def _detect_helmet_color(self, region: np.ndarray) -> dict:
        """
        Detects helmet by HSV color analysis.
        Typical safety helmet colors in mining:
        - Yellow / Orange (most common)
        - White (supervisors)
        - Red (visitors)
        """
        if region.size == 0:
            return {"detected": False, "confidence": 0.0, "color": None}

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        total_pixels = region.shape[0] * region.shape[1]

        color_ranges = {
            "yellow": ([15, 80, 80],  [35, 255, 255]),
            "orange": ([5,  80, 80],  [15, 255, 255]),
            "white":  ([0,  0,  180], [180, 30, 255]),
            "red":    ([0,  100, 100], [5,  255, 255]),
        }

        best_color = None
        best_ratio = 0.0

        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            ratio = cv2.countNonZero(mask) / total_pixels
            if ratio > best_ratio:
                best_ratio = ratio
                best_color = color_name

        # Threshold: at least 8% of the head region must be the helmet color
        detected = best_ratio > 0.08
        confidence = min(0.99, best_ratio * 8)

        return {
            "detected":   detected,
            "confidence": round(confidence, 2),
            "color":      best_color if detected else None
        }

    def _detect_vest_color_from_crop(self, region: np.ndarray) -> dict:
        if region.size == 0:
            return {"detected": False, "confidence": 0.0}
        height = region.shape[0]
        torso = region[int(height * 0.30):int(height * 0.70), :]
        if torso.size == 0:
            torso = region
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        total = torso.shape[0] * torso.shape[1]
        ranges = [
            ([5,  150, 150], [20, 255, 255]),
            ([20, 150, 150], [40, 255, 255]),
        ]
        ratio = sum(
            cv2.countNonZero(cv2.inRange(
                hsv, np.array(lo), np.array(hi))) / total
            for lo, hi in ranges
        )
        return {
            "detected":   ratio > 0.12,
            "confidence": round(min(0.99, ratio * 6), 2)
        }

    def _empty_ppe_result(self) -> dict:
        return {
            "person_confidence": 0.0,
            "helmet": {"detected": False, "confidence": 0.0,
                       "color": None, "method": "empty"},
            "vest":   {"detected": False, "confidence": 0.0},
            "bbox":   {}
        }

# MQTT Callbacks


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        log.info(f"[MQTT] Connected to MQTT broker at {
                 MQTT_BROKER}:{MQTT_PORT}")
    else:
        log.error(f"[MQTT] MQTT connection error, code: {rc}")


def build_event(person_id: int, ppe: dict, frame_num: int) -> dict:
    """
    Builds the standard JSON payload of the system.
    Determines event type and severity based on detected PPE.
    """
    helmet_ok = ppe["helmet"]["detected"]
    vest_ok = ppe["vest"]["detected"]

    # Determine event type and severity
    if not helmet_ok and not vest_ok:
        event_type = "no_helmet_no_vest"
        severity = "critical"
    elif not helmet_ok:
        event_type = "no_helmet"
        severity = "high"
    elif not vest_ok:
        event_type = "no_vest"
        severity = "high"
    else:
        event_type = "ppe_compliant"
        severity = "none"

    # Global confidence: average of detections
    confidence = round(
        (ppe["person_confidence"] +
         ppe["helmet"]["confidence"] +
         ppe["vest"]["confidence"]) / 3,
        2
    )

    return {
        "camera_id":  CAMERA_ID,
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "severity":   severity,
        "confidence": confidence,
        "source":     "yolov8",
        "metadata": {
            "frame":      frame_num,
            "person_id":  person_id,
            "zone":       "entrada-principal",
            "helmet": {
                "detected":   helmet_ok,
                "confidence": ppe["helmet"]["confidence"],
                "color":      ppe["helmet"].get("color")
            },
            "vest": {
                "detected":   vest_ok,
                "confidence": ppe["vest"]["confidence"]
            },
            "bbox": ppe.get("bbox", {})
        }
    }


def run_yolo_detector(client: mqtt.Client):
    detector = EPPDetector(MODEL_PATH, CONFIDENCE_THR)

    log.info("[CAM] Initializing camera (backend=%s)...", CAMERA_BACKEND)
    cam = create_camera(CAMERA_INDEX)
    cam.set_resolution(CAPTURE_WIDTH, CAPTURE_HEIGHT)

    try:
        cam.open()
    except RuntimeError:
        log.error("[CAM] Could not open the camera", exc_info=True)
        return

    log.info(
        "[CAM] Camera ready: %dx%d", CAPTURE_WIDTH, CAPTURE_HEIGHT,
    )
    log.info("[SYS] Publishing to '%s' every %ss", MQTT_TOPIC, INTERVAL_SEC)

    frame_num = 0
    last_publish = 0.0

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                log.warning("[CAM] Dropped frame, retrying...")
                time.sleep(0.5)
                continue

            frame_num += 1
            now = time.time()

            # Publish only every INTERVAL_SEC seconds
            if (now - last_publish) < INTERVAL_SEC:
                continue

            last_publish = now

            persons = detector.detect_persons(frame)

            if not persons:
                # No persons in frame: publish "clear" event
                event = {
                    "camera_id":  CAMERA_ID,
                    "timestamp":  datetime.utcnow().isoformat() + "Z",
                    "event_type": "clear",
                    "severity":   "none",
                    "confidence": 0.99,
                    "source":     "yolov8",
                    "metadata":   {"frame": frame_num, "persons_detected": 0}
                }
                client.publish(MQTT_TOPIC, json.dumps(event), qos=1)
                log.info("[FRAME %d] No persons detected", frame_num)
                continue

            log.info(
                "[FRAME %d] %d person(s) detected", frame_num, len(persons),
            )

            # Analyze PPE for each person
            for person_id, bbox in enumerate(persons):
                ppe = detector.analyze_ppe(frame, bbox)
                event = build_event(person_id, ppe, frame_num)

                client.publish(MQTT_TOPIC, json.dumps(event), qos=1)

                helmet_status = "[CHECK] HELMET" if ppe["helmet"]["detected"] else "NO HELMET"
                vest_status = "[CHECK] VEST" if ppe["vest"]["detected"] else "NO VEST"
                log.info(
                    "  Person %d: %s | %s | event=%s | severity=%s",
                    person_id, helmet_status, vest_status,
                    event["event_type"], event["severity"],
                )

    except KeyboardInterrupt:
        log.info("[SYS] Detector stopped by user")
    finally:
        cam.release()
        log.info("[CAM] Camera released")


def run_simulation(client: mqtt.Client):
    import random
    EVENTS = [
        ("no_helmet",      "high",     0.91),
        ("no_vest",        "high",     0.85),
        ("no_helmet_no_vest", "critical", 0.93),
        ("ppe_compliant",  "none",     0.97),
        ("clear",          "none",     0.99),
    ]
    frame_num = 0
    log.info("[SIM] Simulation mode active (USE_SIMULATION=true)")
    while True:
        frame_num += 1
        event_type, severity, confidence = random.choice(EVENTS)
        event = {
            "camera_id":  CAMERA_ID,
            "timestamp":  datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "severity":   severity,
            "confidence": confidence,
            "source":     "simulated",
            "metadata":   {"frame": frame_num, "zone": "entrada-principal"}
        }
        client.publish(MQTT_TOPIC, json.dumps(event), qos=1)
        log.info("[SIM] %s | severity=%s", event_type, severity)
        time.sleep(INTERVAL_SEC)


def main():
    # Configure MQTT client
    client = mqtt.Client(client_id=f"detector-{CAMERA_ID}")
    client.on_connect = on_connect

    log.info(f"[MQTT] Connecting to {MQTT_BROKER}:{MQTT_PORT}...")
    for attempt in range(10):
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            break
        except Exception as e:
            log.warning(f"[MQTT] Attempt {attempt+1}/10 failed: {e}")
            time.sleep(3)
    else:
        log.error("[MQTT] Could not connect. Aborting.")
        return

    client.loop_start()

    if USE_SIMULATION:
        run_simulation(client)
    else:
        run_yolo_detector(client)

    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()
