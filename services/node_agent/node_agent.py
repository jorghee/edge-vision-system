"""
Node Agent: Lightweight device metrics collector for Edge Vision System.

Collects system metrics (CPU, RAM, temperature, disk, network) and
eKuiper inference metrics, then publishes them via MQTT to the central
server for Grafana visualization.

Designed to run on Raspberry Pi with minimal resource usage (<15MB RAM).
"""

import json
import os
import time
import logging
import subprocess
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [NODE_AGENT] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    psutil = None
    log.warning("psutil not available, system metrics will be limited")

try:
    import paho.mqtt.client as mqtt
except ImportError:
    log.error("paho-mqtt is required")
    raise

# Configuration
MQTT_SERVER_URL = os.getenv("MQTT_SERVER_URL", "tcp://localhost:1883")
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")
CAMERA_ID = os.getenv("CAMERA_ID", "cam-rpi-01")
COLLECT_INTERVAL = int(os.getenv("COLLECT_INTERVAL", "30"))
EKUIPER_API = os.getenv("EKUIPER_API", "http://ekuiper-engine:9081")
METRICS_TOPIC = f"edge/metrics/{CAMERA_ID}"


def parse_mqtt_url(url: str) -> tuple:
    """Parse tcp://host:port into (host, port)."""
    url = url.replace("tcp://", "").replace("ssl://", "")
    parts = url.split(":")
    host = parts[0]
    port = int(parts[1]) if len(parts) > 1 else 1883
    return host, port


def get_system_metrics() -> dict:
    """Collect system-level metrics."""
    metrics = {}

    if psutil:
        metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        metrics["memory_percent"] = round(mem.percent, 1)
        metrics["memory_used_mb"] = round(mem.used / (1024 * 1024), 1)
        disk = psutil.disk_usage("/")
        metrics["disk_percent"] = round(disk.percent, 1)

        net = psutil.net_io_counters()
        metrics["net_bytes_tx"] = net.bytes_sent
        metrics["net_bytes_rx"] = net.bytes_recv

        metrics["uptime_seconds"] = int(time.time() - psutil.boot_time())

    # RPi temperature (via /sys/class/thermal or vcgencmd)
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_milli = int(f.read().strip())
            metrics["temperature_c"] = round(temp_milli / 1000.0, 1)
    except (FileNotFoundError, ValueError, PermissionError):
        try:
            result = subprocess.run(
                ["vcgencmd", "measure_temp"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # Output: temp=42.8'C
                temp_str = result.stdout.strip().replace("temp=", "").replace("'C", "")
                metrics["temperature_c"] = float(temp_str)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return metrics


def get_container_metrics() -> dict:
    """Check Docker container status via the Docker socket."""
    metrics = {"containers_total": 0, "containers_running": 0}

    try:
        import urllib.request
        req = urllib.request.Request(
            "http://localhost/containers/json?all=true"
        )
        # Connect via Unix socket
        import http.client
        conn = http.client.HTTPConnection("localhost")
        conn.sock = __import__("socket").socket(
            __import__("socket").AF_UNIX,
            __import__("socket").SOCK_STREAM
        )
        conn.sock.connect("/var/run/docker.sock")
        conn.request("GET", "/containers/json?all=true")
        resp = conn.getresponse()
        containers = json.loads(resp.read().decode())
        conn.close()

        metrics["containers_total"] = len(containers)
        metrics["containers_running"] = sum(
            1 for c in containers if c.get("State") == "running"
        )

        # Check specific services
        for c in containers:
            names = [n.strip("/") for n in c.get("Names", [])]
            if "ekuiper-engine" in names:
                metrics["ekuiper_status"] = c.get("State", "unknown")
            if "mediamtx" in names:
                metrics["mediamtx_status"] = c.get("State", "unknown")

    except Exception as e:
        log.debug("Could not read Docker socket: %s", e)

    return metrics


def get_ekuiper_metrics() -> dict:
    """Query eKuiper REST API for rule processing metrics."""
    metrics = {}

    try:
        import urllib.request
        rules = ["ppe_alert_critical", "ppe_alert_high", "ppe_monitor"]

        total_in = 0
        total_errors = 0
        max_latency = 0

        for rule_id in rules:
            url = f"{EKUIPER_API}/rules/{rule_id}/status"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())

            if data.get("status") != "running":
                continue

            # Look for source metrics (records_in_total, process_latency)
            for key, value in data.items():
                if key.endswith("records_in_total") and "source" in key:
                    total_in += value
                elif key.endswith("process_latency_us") and "source" in key:
                    max_latency = max(max_latency, value)
                elif key.endswith("exceptions_total") and "source" in key:
                    total_errors += value

        metrics["frames_processed"] = total_in
        metrics["inference_latency_ms"] = round(max_latency / 1000.0, 1)
        metrics["inference_errors"] = total_errors

    except Exception as e:
        log.debug("Could not query eKuiper API: %s", e)

    return metrics


def collect_all_metrics() -> dict:
    """Aggregate all metrics into a single payload."""
    payload = {
        "camera_id": CAMERA_ID,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    payload.update(get_system_metrics())
    payload.update(get_container_metrics())
    payload.update(get_ekuiper_metrics())
    return payload


def main():
    host, port = parse_mqtt_url(MQTT_SERVER_URL)

    client = mqtt.Client(client_id=f"node-agent-{CAMERA_ID}")
    if MQTT_USERNAME:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    # Connect with retry
    for attempt in range(30):
        try:
            client.connect(host, port, keepalive=60)
            client.loop_start()
            log.info("Connected to MQTT broker at %s:%d", host, port)
            break
        except Exception as e:
            log.warning("MQTT connection attempt %d/30: %s", attempt + 1, e)
            time.sleep(5)

    log.info(
        "Node Agent started. Publishing to '%s' every %ds",
        METRICS_TOPIC, COLLECT_INTERVAL,
    )

    try:
        while True:
            try:
                metrics = collect_all_metrics()
                payload = json.dumps(metrics)
                result = client.publish(METRICS_TOPIC, payload, qos=0)
                if result.rc == 0:
                    log.info(
                        "Published metrics: CPU=%.1f%% MEM=%.1f%% TEMP=%.1f°C",
                        metrics.get("cpu_percent", 0),
                        metrics.get("memory_percent", 0),
                        metrics.get("temperature_c", 0),
                    )
                else:
                    log.warning("Publish failed: rc=%d", result.rc)
            except Exception as e:
                log.error("Error collecting/publishing metrics: %s", e)

            time.sleep(COLLECT_INTERVAL)

    except KeyboardInterrupt:
        log.info("Node Agent stopped")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
