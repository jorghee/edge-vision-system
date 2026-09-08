#!/bin/sh
# Generates the Mosquitto password file on first start.
# Credentials are read from environment variables.

PASSWD_FILE="/mosquitto/config/passwd"

if [ ! -f "$PASSWD_FILE" ]; then
    echo "Creating MQTT password file..."
    touch "$PASSWD_FILE"
    # Edge device credentials (eKuiper on RPi)
    mosquitto_passwd -b "$PASSWD_FILE" "${MQTT_USER_EDGE:-edge_device}" "${MQTT_PASS_EDGE:-SecureEdge2026!}"
    # Action service credentials
    mosquitto_passwd -b "$PASSWD_FILE" "${MQTT_USER_ACTION:-action_service}" "${MQTT_PASS_ACTION:-SecureAction2026!}"
    echo "Password file created with users: edge_device, action_service"
else
    echo "Password file already exists, skipping generation."
fi

# Start Mosquitto
exec mosquitto -c /mosquitto/config/mosquitto.conf
