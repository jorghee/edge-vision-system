# Despliegue y Configuración

El despliegue de este sistema debe abordarse en dos fases físicas diferenciadas: primero se debe aprovisionar el **Servidor Central** (para que exista un destino hacia donde enviar la información), y posteriormente se despliegan los **Dispositivos Edge**.

Todas las dependencias están empaquetadas en contenedores Docker, garantizando un despliegue predecible independientemente del sistema operativo base.

## 1. Servidor Central

El Servidor Central puede ser una máquina local, un servidor on-premise, o una instancia en la nube (AWS EC2, Google Cloud Compute Engine). Solo requiere Linux y Docker.

### Componentes que levanta:
- Broker MQTT (Mosquitto)
- Base de datos (InfluxDB)
- Colector (Telegraf)
- Dashboards (Grafana)

### Pasos de Despliegue:

1. Ingresar al directorio del servidor central:
   ```bash
   cd infrastructure/central-server
   ```
2. (Opcional) Ajustar el archivo `.env` si se requiere modificar los puertos por defecto, el token de InfluxDB, o las credenciales de Grafana.
3. Levantar la infraestructura en segundo plano:
   ```bash
   docker compose -f docker-compose.server.yml up -d
   ```
4. Verificar que todos los servicios estén operacionales:
   ```bash
   docker ps
   ```

A partir de este momento, Grafana está disponible en `http://<IP_DEL_SERVIDOR>:3000` (usuario `admin`, clave `admin` por defecto) y el Broker MQTT está esperando conexiones en el puerto `1883`.

---

## 2. Dispositivo Edge (Ej. Raspberry Pi)

El dispositivo en el borde requiere tener conectada físicamente una cámara (ya sea por USB o módulo CSI compatible con V4L2) y acceso a red para alcanzar al Servidor Central.

> [!IMPORTANT]
> Antes de desplegar el Edge Device, asegúrese de tener la dirección IP o dominio del Servidor Central.

### Componentes que levanta:
- Servidor RTSP Local (MediaMTX)
- Motor de Procesamiento y Reglas (eKuiper)
- Script de Telemetría (Node Agent)

### Pasos de Despliegue (Automatizado mediante Script):

El proyecto incluye un script robusto que automatiza la instalación completa. Copie el repositorio a la Raspberry Pi y ejecute:

1. Ejecutar el script principal indicando la URL MQTT del servidor central:
   ```bash
   bash scripts/deploy_edge.sh "tcp://<IP_DEL_SERVIDOR_CENTRAL>:1883"
   ```

### ¿Qué hace el script `deploy_edge.sh` por debajo?

Si desea desplegar manualmente o requiere auditar el script, estos son los pasos internos que ejecuta:

1. **Gestión de variables:** Inserta la IP del Servidor Central en el archivo `infrastructure/edge-device/.env` bajo la variable `MQTT_SERVER_URL`.
2. **Levantamiento Docker:** Ejecuta `docker compose -f infrastructure/edge-device/docker-compose.device.yml up -d` para iniciar MediaMTX, eKuiper y el Node Agent.
3. **Conversión de Modelos:** Si eKuiper arranca correctamente, ejecuta `scripts/prepare_models.sh` para asegurar que el modelo YOLOv8 crudo (.pt) se convierta y esté disponible en formato TFLite o NCNN (requeridos para eKuiper).
4. **Aprovisionamiento de eKuiper:** Ejecuta `scripts/setup_ekuiper.sh`, el cual interactúa con la API REST de eKuiper para inyectar, en este orden:
   - El plugin portable en Python (`ppe_inference`).
   - El stream origen conectado a MediaMTX (`camera_frames`).
   - Las reglas SQL que vinculan el stream con el plugin y configuran las salidas hacia MQTT.

> [!WARNING]
> La cámara física debe estar disponible en la ruta `/dev/video0`. Si su cámara se monta en un path distinto (ej. `/dev/video2`), modifique el valor `CAM_DEVICE` en el archivo `infrastructure/edge-device/.env` antes de ejecutar el script.

---

## 3. Simulación Local (Modo Desarrollo)

Si usted no posee una Raspberry Pi y desea **simular el sistema completo** directamente en su Laptop (x86_64) para motivos de desarrollo o depuración, puede desplegar ambos lados simultáneamente.

1. Ejecute el script de simulación:
   ```bash
   bash scripts/start_simulation.sh
   ```
Este script levantará una arquitectura combinada (`infrastructure/local-simulation`) que fusiona los componentes centrales y los del borde en una única red de Docker local, utilizando la webcam conectada a su computadora como origen de datos.
