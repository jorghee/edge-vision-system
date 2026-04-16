FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY detector.py .

ENV MQTT_BROKER=mqtt
ENV MQTT_PORT=1883
ENV MQTT_TOPIC=camera/events
ENV CAMERA_ID=cam-01
ENV INTERVAL_SEC=3

CMD ["python3", "detector.py"]
