FROM python:3.11-slim

# OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/download_model.py .
RUN python3 download_model.py

COPY src/camera.py .
COPY src/detector.py .
COPY src/health_monitor.py .
COPY scripts/export_model.py .

ENV MQTT_BROKER=mqtt
ENV MQTT_PORT=1883
ENV MQTT_TOPIC=camera/events
ENV CAMERA_ID=cam-01
ENV CAMERA_INDEX=2
ENV CAMERA_BACKEND=opencv
ENV INTERVAL_SEC=3
ENV USE_SIMULATION=false
ENV MODEL_PATH=/app/models/yolov8n.pt
ENV CONFIDENCE_THR=0.45
ENV CAPTURE_WIDTH=640
ENV CAPTURE_HEIGHT=480

CMD ["python3", "detector.py"]
