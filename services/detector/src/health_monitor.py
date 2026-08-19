"""
Hardware health monitor for Raspberry Pi.

Publishes system metrics (CPU temp, CPU usage, RAM usage)
to the MQTT broker on the `edge/health` topic every `HEALTH_INTERVAL` seconds.

Designed to run as a companion process to the detector on the RPi.
The published metrics can be consumed by eKuiper rules to
generate alerts for overheating or resource exhaustion.

Usage:
    python health_monitor.py

Environment variables:
    MQTT_BROKER       - MQTT broker host (default: localhost)
    MQTT_PORT         - MQTT port (default: 1883)
    HEALTH_TOPIC      - Publish topic (default: edge/health)
    HEALTH_INTERVAL   - Seconds between publications (default: 30)
    CAMERA_ID         - Device identifier (default: cam-rpi-01)
"""

import json
import logging
import os
import time
from datetime import datetime

import paho.mqtt.client as mqtt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [HEALTH] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
HEALTH_TOPIC = os.getenv("HEALTH_TOPIC", "edge/health")
HEALTH_INTERVAL = int(os.getenv("HEALTH_INTERVAL", "30"))
CAMERA_ID = os.getenv("CAMERA_ID", "cam-rpi-01")


def get_cpu_temp() -> float:
    """Reads CPU temperature from sysfs (°C).

    Available on Raspberry Pi and most Linux SBCs.
    Returns -1.0 if it cannot be read.
    """
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read().strip()) / 1000.0, 1)
    except (FileNotFoundError, ValueError):
        return -1.0


def get_cpu_usage_pct() -> float:
    """CPU usage as a percentage based on load average (1 min)."""
    try:
        load_1min = os.getloadavg()[0]
        cores = os.cpu_count() or 4
        return round(load_1min / cores * 100.0, 1)
    except OSError:
        return -1.0


def get_memory_info() -> dict:
    """Returns RAM usage in MB and percentage reading /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()

        meminfo = {}
        for line in lines[:5]:
            parts = line.split()
            meminfo[parts[0].rstrip(":")] = int(parts[1])

        total_mb = meminfo["MemTotal"] / 1024.0
        available_mb = meminfo.get("MemAvailable", meminfo["MemFree"]) / 1024.0
        used_mb = total_mb - available_mb

        return {
            "total_mb": round(total_mb),
            "used_mb": round(used_mb),
            "available_mb": round(available_mb),
            "usage_pct": round(used_mb / total_mb * 100.0, 1),
        }
    except (FileNotFoundError, KeyError, ValueError):
        return {"total_mb": -1, "used_mb": -1, "available_mb": -1, "usage_pct": -1}


def get_disk_usage() -> dict:
    """Disk usage of the root partition."""
    try:
        stat = os.statvfs("/")
        total = stat.f_blocks * stat.f_frsize
        free = stat.f_bfree * stat.f_frsize
        used = total - free
        return {
            "total_gb": round(total / 1e9, 1),
            "used_gb": round(used / 1e9, 1),
            "usage_pct": round(used / total * 100.0, 1) if total else -1,
        }
    except OSError:
        return {"total_gb": -1, "used_gb": -1, "usage_pct": -1}


def get_throttle_status() -> dict:
    """Reads the throttling status of the RPi (vcgencmd).

    Relevant flags:
        bit 0: Under-voltage detected
        bit 1: Arm frequency capped
        bit 2: Currently throttled
        bit 3: Soft temperature limit active
    """
    try:
        import subprocess

        result = subprocess.run(
            ["vcgencmd", "get_throttled"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Output: throttled=0x0
        hex_val = result.stdout.strip().split("=")[-1]
        flags = int(hex_val, 16)
        return {
            "raw": hex_val,
            "under_voltage": bool(flags & 0x1),
            "freq_capped": bool(flags & 0x2),
            "throttled": bool(flags & 0x4),
            "soft_temp_limit": bool(flags & 0x8),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return {"raw": "unavailable"}


def build_health_payload() -> dict:
    """Builds the complete JSON payload of health metrics."""
    mem = get_memory_info()
    disk = get_disk_usage()
    throttle = get_throttle_status()

    return {
        "device_id": CAMERA_ID,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "cpu": {
            "temp_celsius": get_cpu_temp(),
            "usage_pct": get_cpu_usage_pct(),
        },
        "memory": mem,
        "disk": disk,
        "throttle": throttle,
    }


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        log.info("Connected to MQTT broker at %s:%d", MQTT_BROKER, MQTT_PORT)
    else:
        log.error("MQTT connection error, code: %d", rc)


def main() -> None:
    client = mqtt.Client(client_id=f"health-{CAMERA_ID}")
    client.on_connect = on_connect

    log.info("Connecting to %s:%d...", MQTT_BROKER, MQTT_PORT)
    for attempt in range(10):
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            break
        except Exception as e:
            log.warning("Attempt %d/10 failed: %s", attempt + 1, e)
            time.sleep(3)
    else:
        log.error("Could not connect to MQTT broker. Aborting.")
        return

    client.loop_start()

    log.info(
        "Publishing metrics to '%s' every %ds", HEALTH_TOPIC, HEALTH_INTERVAL,
    )

    try:
        while True:
            payload = build_health_payload()
            client.publish(HEALTH_TOPIC, json.dumps(payload), qos=0)

            temp = payload["cpu"]["temp_celsius"]
            cpu = payload["cpu"]["usage_pct"]
            mem = payload["memory"]["usage_pct"]
            log.info(
                "CPU: %.1f°C (%s%%) | RAM: %s%% | Disk: %s%%",
                temp, cpu, mem, payload["disk"]["usage_pct"],
            )

            # Temperature alerts in logs
            if temp > 80:
                log.warning("ALERT: High CPU temperature: %.1f°C", temp)
            if temp > 85:
                log.critical(
                    "CRITICAL: Dangerous CPU temperature: %.1f°C — risk of severe throttling",
                    temp,
                )

            time.sleep(HEALTH_INTERVAL)

    except KeyboardInterrupt:
        log.info("Monitor stopped by user")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
