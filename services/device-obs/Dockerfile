FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir psutil paho-mqtt

COPY node_agent.py .

CMD ["python3", "-u", "node_agent.py"]
