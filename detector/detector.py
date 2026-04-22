import json
import time
import os
import logging
from datetime import datetime

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DETECTOR] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)

import ultralytics
ultralytics.checks = lambda: None
logging.getLogger("ultralytics").setLevel(logging.WARNING)

MQTT_BROKER   = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT     = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC    = os.getenv("MQTT_TOPIC", "camera/events")
CAMERA_ID     = os.getenv("CAMERA_ID", "cam-01")
CAMERA_INDEX  = int(os.getenv("CAMERA_INDEX", "2"))
INTERVAL_SEC  = float(os.getenv("INTERVAL_SEC", "3"))
USE_SIMULATION = os.getenv("USE_SIMULATION", "false").lower() == "true"
MODEL_PATH     = os.getenv("MODEL_PATH", "/app/models/yolov8n.pt")
CONFIDENCE_THR = float(os.getenv("CONFIDENCE_THR", "0.45"))

class EPPDetector:
    """
    Detector de Equipos de Protección Personal usando YOLOv8.

    El modelo yolov8n.pt fue entrenado en COCO y reconoce 80 clases.
    La clase 0 es 'person'. Usamos esa detección como primer filtro.

    Para el EPP usamos análisis de región sobre cada persona detectada:
    - Región superior (30% de altura): cabeza, detectar casco
    - Región media  (40% de altura): torso, detectar chaleco
    """

    # Clases COCO relevantes
    COCO_PERSON_CLASS = 0

    EPP_CLASSES = {
        "helmet":     0,   # casco detectado
        "head":       1,   # cabeza sin casco
        "person":     2,   # persona (redundante con modelo base)
    }

    def __init__(self, model_path: str, confidence: float = 0.45):
        ppe_path = os.path.join(os.path.dirname(model_path), "ppe_detector.pt")

        log.info(f"[YOLO] Cargando modelo base desde {model_path}...")
        self.model_base = YOLO(model_path)

        # Cargar modelo EPP si existe
        if os.path.exists(ppe_path):
            log.info(f"[YOLO] Cargando modelo EPP desde {ppe_path}...")
            self.model_ppe     = YOLO(ppe_path)
            self.use_ppe_model = True
            log.info("[YOLO] Modelo EPP cargado — inferencia real de casco")
        else:
            log.warning("[YOLO] Modelo EPP no encontrado, usando análisis de color")
            self.model_ppe     = None
            self.use_ppe_model = False

        self.confidence = confidence
        log.info("[YOLO] Modelos listos")

    def detect_persons(self, frame: np.ndarray) -> list:
        """
        Ejecuta YOLOv8 sobre el frame completo.
        Retorna lista de bounding boxes de personas detectadas.
        Cada bbox es (x1, y1, x2, y2, confidence).
        """
        results = self.model_base(
            frame,
            conf=self.confidence,
            classes=[self.COCO_PERSON_CLASS],  # Solo detectar personas
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
        Analiza EPP de una persona.
        Usa modelo EPP si está disponible, color HSV como fallback.
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
        Inferencia real con modelo fine-tuned.
        Corre localmente en CPU — sin internet, sin nube.
        """
        x1, y1, x2, y2, _ = bbox
        results = self.model_ppe(crop, conf=self.confidence, verbose=False)

        helmet_detected    = False
        helmet_confidence  = 0.0
        head_detected      = False   # cabeza sin casco

        for result in results:
            for box in result.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == self.EPP_CLASSES["helmet"]:
                    helmet_detected   = True
                    helmet_confidence = max(helmet_confidence, conf)

                elif cls == self.EPP_CLASSES["head"]:
                    head_detected = True   # confirma que hay cabeza visible

        # Si detectó cabeza pero no casco → no_helmet confirmado
        # Si no detectó ni cabeza ni casco → persona de espaldas, ignorar
        if not head_detected and not helmet_detected:
            # No se puede determinar — asumir compliant para no generar falsos
            helmet_detected   = True
            helmet_confidence = 0.5

        return {
            "person_confidence": round(person_conf, 2),
            "helmet": {
                "detected":   helmet_detected,
                "confidence": round(helmet_confidence, 2),
                "color":      None,
                "method":     "yolo_model"
            },
            # Este modelo no detecta chaleco — usar color como complemento
            "vest": self._detect_vest_color_from_crop(crop),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        }

    def _analyze_with_color(self, crop: np.ndarray, person_conf: float, bbox: tuple) -> dict:
        """Fallback: análisis de color HSV cuando no hay modelo EPP."""
        x1, y1, x2, y2, _ = bbox
        height = crop.shape[0]

        head_region  = crop[0:int(height * 0.30), :]
        torso_region = crop[int(height * 0.30):int(height * 0.70), :]

        return {
            "person_confidence": round(person_conf, 2),
            "helmet": {**self._detect_helmet_color(head_region), "method": "color_hsv"},
            "vest":   self._detect_vest_color_from_crop(torso_region),
            "bbox":   {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        }

    def _detect_helmet_color(self, region: np.ndarray) -> dict:
        """
        Detecta casco por análisis de color en HSV.
        Colores típicos de cascos de seguridad en minería:
        - Amarillo / Naranja (más comunes)
        - Blanco (supervisores)
        - Rojo (visitantes)
        """
        if region.size == 0:
            return {"detected": False, "confidence": 0.0, "color": None}

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        total_pixels = region.shape[0] * region.shape[1]

        color_ranges = {
            "yellow": ([15, 80, 80],  [35, 255, 255]),
            "orange": ([5,  80, 80],  [15, 255, 255]),
            "white":  ([0,  0,  180], [180, 30, 255]),
            "red":    ([0,  100, 100],[5,  255, 255]),
        }

        best_color    = None
        best_ratio    = 0.0

        for color_name, (lower, upper) in color_ranges.items():
            mask  = cv2.inRange(hsv, np.array(lower), np.array(upper))
            ratio = cv2.countNonZero(mask) / total_pixels
            if ratio > best_ratio:
                best_ratio = ratio
                best_color = color_name

        # Umbral: al menos 8% de la región de cabeza debe ser del color del casco
        detected   = best_ratio > 0.08
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
        torso  = region[int(height * 0.30):int(height * 0.70), :]
        if torso.size == 0:
            torso = region
        hsv   = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        total = torso.shape[0] * torso.shape[1]
        ranges = [
            ([5,  150, 150], [20, 255, 255]),
            ([20, 150, 150], [40, 255, 255]),
        ]
        ratio = sum(
            cv2.countNonZero(cv2.inRange(hsv, np.array(lo), np.array(hi))) / total
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
        log.info(f"[MQTT] Conectado al broker MQTT en {MQTT_BROKER}:{MQTT_PORT}")
    else:
        log.error(f"[MQTT] Error de conexión MQTT, código: {rc}")

def build_event(person_id: int, ppe: dict, frame_num: int) -> dict:
    """
    Construye el payload JSON estándar del sistema.
    Determina el tipo de evento y severidad según el EPP detectado.
    """
    helmet_ok = ppe["helmet"]["detected"]
    vest_ok   = ppe["vest"]["detected"]

    # Determinar tipo de evento y severidad
    if not helmet_ok and not vest_ok:
        event_type = "no_helmet_no_vest"
        severity   = "critical"
    elif not helmet_ok:
        event_type = "no_helmet"
        severity   = "high"
    elif not vest_ok:
        event_type = "no_vest"
        severity   = "high"
    else:
        event_type = "ppe_compliant"
        severity   = "none"

    # Confianza global: promedio de las detecciones
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

    log.info(f"[CAM] Abriendo dispositivo {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        log.error(f"[CAM] No se pudo abrir {CAMERA_INDEX}")
        return

    # Configurar resolución (reducir carga de CPU)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    log.info(f"[CAM] Cámara lista: 640x480 @ 15fps")
    log.info(f"[SYS] Publicando en '{MQTT_TOPIC}' cada {INTERVAL_SEC}s")

    frame_num      = 0
    last_publish   = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("[CAM] Frame perdido, reintentando...")
                time.sleep(0.5)
                continue

            frame_num += 1
            now = time.time()

            # Publicar solo cada INTERVAL_SEC segundos
            if (now - last_publish) < INTERVAL_SEC:
                continue

            last_publish = now

            persons = detector.detect_persons(frame)

            if not persons:
                # Sin personas en el frame: publicar evento "clear"
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
                log.info(f"[FRAME {frame_num}] Sin personas detectadas")
                continue

            log.info(f"[FRAME {frame_num}] {len(persons)} persona(s) detectada(s)")

            # Analizar EPP de cada persona
            for person_id, bbox in enumerate(persons):
                ppe   = detector.analyze_ppe(frame, bbox)
                event = build_event(person_id, ppe, frame_num)

                client.publish(MQTT_TOPIC, json.dumps(event), qos=1)

                helmet_status = "[CHECK] CASCO" if ppe["helmet"]["detected"] else "❌ SIN CASCO"
                vest_status   = "[CHECK] CHALECO" if ppe["vest"]["detected"] else "❌ SIN CHALECO"
                log.info(
                    f"  Persona {person_id}: {helmet_status} | {vest_status} "
                    f"| evento={event['event_type']} | severidad={event['severity']}"
                )

    except KeyboardInterrupt:
        log.info("[SYS] Detector detenido por el usuario")
    finally:
        cap.release()
        log.info("[CAM] Cámara liberada")

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
    log.info("[SIM] Modo simulación activo (USE_SIMULATION=true)")
    while True:
        frame_num += 1
        event_type, confidence = random.choice(EVENTS)
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
        log.info(f"[SIM] {event_type} | severidad={confidence}")
        time.sleep(INTERVAL_SEC)

def main():
    # Configure MQTT client
    client = mqtt.Client(client_id=f"detector-{CAMERA_ID}")
    client.on_connect = on_connect

    log.info(f"[MQTT] Conectando a {MQTT_BROKER}:{MQTT_PORT}...")
    for attempt in range(10):
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            break
        except Exception as e:
            log.warning(f"[MQTT] Intento {attempt+1}/10 fallido: {e}")
            time.sleep(3)
    else:
        log.error("[MQTT] No se pudo conectar. Abortando.")
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
