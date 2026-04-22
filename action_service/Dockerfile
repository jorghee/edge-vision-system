FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY action_service.py .

ENV MQTT_BROKER=mqtt
ENV MQTT_PORT=1883
ENV ALERT_TOPIC=edge/alerts
ENV ACTION_TOPIC=edge/actions

CMD ["python3", "action_service.py"]
