# Edge Vision System

Bienvenido al **Edge Vision System**, una solución integral de visión artificial y procesamiento de eventos diseñada específicamente para entornos de **Edge Computing** (computación en el borde). 

## Presentación del Proyecto
En entornos industriales y mineros, la transmisión constante de video en alta resolución hacia servidores centrales o la nube resulta inviable debido a limitaciones de ancho de banda, latencias elevadas y altos costos operativos. El **Edge Vision System** aborda este desafío procesando el flujo de video de manera local en el propio dispositivo perimetral (ej. una Raspberry Pi) utilizando redes neuronales (YOLOv8). 

El sistema detecta eventos relevantes en tiempo real —como la presencia de personal y el cumplimiento del uso de Equipos de Protección Personal (EPP)— y transmite únicamente **alertas críticas estructuradas**. Esto elimina la transmisión de datos innecesarios hacia el exterior, asegurando una respuesta inmediata ante incidentes.

## Propuesta de Valor
Este proyecto demuestra la implementación práctica de una arquitectura distribuida eficiente:
* **IoT y Edge Computing:** El procesamiento intensivo se realiza en la frontera de la red, reduciendo la latencia de análisis a milisegundos.
* **Eficiencia de Ancho de Banda:** En lugar de enviar un flujo continuo de video (MB/s), el sistema transmite payloads JSON ligeros (KB/s) únicamente cuando ocurre un evento de interés.
* **Filtrado Inteligente:** Incorpora un motor de reglas perimetral capaz de descartar telemetría normal y priorizar notificaciones críticas de forma autónoma.
* **Despliegue Versátil:** Preparado para funcionar en servidores estándar (x86_64) y optimizado para dispositivos IoT de recursos limitados (ARM64) mediante la aceleración nativa de modelos matemáticos.

## Arquitectura General
El sistema está construido bajo un enfoque de microservicios orientados a eventos, totalmente desacoplados a través de un bus MQTT.

* **Detector de Visión:** Componente en Python que captura el video, ejecuta inferencia local y clasifica la escena.
* **Broker MQTT (Mosquitto):** Eje central de mensajería asíncrona y de muy baja latencia.
* **Motor de Reglas (LF Edge eKuiper):** Analiza el flujo de eventos continuos (*stream*), aplica reglas SQL declarativas y filtra el ruido.
* **Action Service:** Servicio reactivo que escucha las alertas depuradas para integrarlas con lógicas operativas (notificaciones, logs, actuadores).

> [!NOTE]
> Para una inmersión técnica profunda, consulta el documento [Arquitectura y Flujo de Datos](docs/architecture.md).

## Características Principales (Implementadas)
* **Detección de Personas:** Identificación espacial en tiempo real utilizando la arquitectura YOLOv8.
* **Análisis de EPP:** Verificación del uso de casco y chaleco de seguridad mediante una estrategia dual (modelos *fine-tuned* y fallback probabilístico basado en segmentación de color HSV).
* **Abstracción de Hardware:** Soporte transparente para cámaras USB tradicionales (OpenCV) y módulos CSI nativos de hardware embebido (Picamera2/libcamera).
* **Monitoreo de Salud:** Agente de telemetría del dispositivo físico que reporta uso de CPU, estado de la memoria RAM, estrés térmico y eventos de *throttling*.
* **Aceleración ARM (NCNN):** Herramientas propias de exportación neuronal para aprovechar las instrucciones SIMD (NEON) de procesadores móviles.

## Tecnologías Utilizadas
* **Python 3.11:** Lógica central para integración de hardware, manipulación de video y orquestación de eventos.
* **Ultralytics YOLOv8 & OpenCV:** Frameworks core para el pipeline de inferencia y procesamiento matricial.
* **NCNN:** Motor de inferencia de alto rendimiento optimizado para arquitecturas ARM.
* **Eclipse Mosquitto:** Bus IoT estándar en la industria por su huella mínima y alta confiabilidad.
* **LF Edge eKuiper:** Procesamiento de flujos de datos y analítica (Stream Processing) nativo para el Edge.
* **Docker & Docker Compose:** Contenerización estandarizada para garantizar portabilidad de la infraestructura.

## Flujo de Funcionamiento
1. El **Detector** lee los frames de la cámara y los procesa con inteligencia artificial de manera continua.
2. Al identificar a un trabajador, verifica el cumplimiento de su EPP, empaqueta los resultados y asigna una severidad (ej. `critical` si carece del equipo obligatorio).
3. Publica un payload JSON estandarizado en el topic continuo `camera/events`.
4. **eKuiper** intercepta este volumen de datos. Si su evaluación SQL determina una criticidad, republica el evento en el topic limpio `edge/alerts`.
5. El **Action Service** recibe únicamente la alerta validada, lográndola formalmente y desplegando las recomendaciones de mitigación pertinentes en `edge/actions`.

> [!TIP]
> Descubre más sobre las estrategias visuales y algoritmos implementados en [Sistema de Detección](docs/detector.md).

## Aplicaciones Potenciales en la Industria
El diseño modular del Edge Vision System permite su integración en múltiples escenarios de transformación digital y automatización de la industria minera o pesada:
* **Seguridad Proactiva (HSE):** Control de acceso automatizado a zonas de alto riesgo, validando activamente que los operarios porten el equipamiento reglamentario.
* **Monitoreo en Maquinaria:** Adaptación del módulo de visión como nodo perimetral instalado en maquinaria pesada para advertir sobre personal ingresando en zonas de punto ciego.
* **Integración con SCADA/PLCs:** Transformación de alertas JSON generadas por el *Action Service* en señales eléctricas para sistemas de paro de emergencia o balizas luminosas en planta.
* **Nodos Descentralizados:** Agrupación de decenas de cámaras de procesamiento autónomo que reportan incidencias a un centro de control remoto (dashboard) sin colapsar el ancho de banda corporativo.

*(Nota: Estas descripciones ilustran el alcance del diseño distribuido del proyecto. La adaptación a hardware de campo específico o conectores industriales requeriría configuraciones complementarias a las actualmente implementadas).*

---

## Inicio Rápido

El proyecto está preparado para inicializarse rápidamente en un entorno unificado con Docker.

### 1. Iniciar los Servicios
Para compilar las imágenes y levantar los contenedores:
```bash
docker-compose up --build -d
```
*Si tu objetivo es desplegar en dispositivos IoT ARM como la Raspberry Pi, es fundamental revisar [Configuración y Despliegue](docs/deployment.md) para emplear el modelo híbrido.*

### 2. Configurar el Motor de Reglas (eKuiper)
Una vez en ejecución, eKuiper debe ser aprovisionado con sus streams y reglas de filtrado mediante su API REST:

<details>
<summary><b>Mostrar comandos de configuración (Limpieza, Stream y Reglas)</b></summary>

**Limpiar reglas anteriores (opcional):**
```bash
curl -X DELETE http://localhost:9081/rules/alert_critical
curl -X DELETE http://localhost:9081/rules/alert_high
curl -X DELETE http://localhost:9081/rules/monitor_all
curl -X DELETE http://localhost:9081/streams/camera_events
```

**Definir el Stream de entrada:**
```bash
curl -X POST http://localhost:9081/streams \
  -H "Content-Type: application/json" \
  -d '{
    "sql": "CREATE STREAM camera_events() WITH (DATASOURCE=\"camera/events\", FORMAT=\"json\", TYPE=\"mqtt\", CONF_KEY=\"default\")"
  }'
```

**Regla 1: Alertas Críticas (ej. sin ningún EPP):**
```bash
curl -X POST http://localhost:9081/rules \
  -H "Content-Type: application/json" \
  -d '{
    "id": "alert_critical",
    "sql": "SELECT camera_id, event_type, severity, confidence, timestamp, metadata FROM camera_events WHERE severity = '\''critical'\''",
    "actions": [
      { "mqtt": { "server": "tcp://mqtt:1883", "topic": "edge/alerts", "qos": 1 } },
      { "log": {} }
    ]
  }'
```

**Regla 2: Alertas Altas (ej. falta casco o chaleco):**
```bash
curl -X POST http://localhost:9081/rules \
  -H "Content-Type: application/json" \
  -d '{
    "id": "alert_high",
    "sql": "SELECT camera_id, event_type, severity, confidence, timestamp, metadata FROM camera_events WHERE severity = '\''high'\''",
    "actions": [
      { "mqtt": { "server": "tcp://mqtt:1883", "topic": "edge/alerts", "qos": 1 } },
      { "log": {} }
    ]
  }'
```

**Regla 3: Monitoreo Básico (Bitácora de zonas ocupadas):**
```bash
curl -X POST http://localhost:9081/rules \
  -H "Content-Type: application/json" \
  -d '{
    "id": "monitor_all",
    "sql": "SELECT camera_id, event_type, severity, confidence, timestamp FROM camera_events WHERE event_type != '\''clear'\''",
    "actions": [
      { "mqtt": { "server": "tcp://mqtt:1883", "topic": "edge/monitor", "qos": 0 } }
    ]
  }'
```
</details>

### 3. Comandos Esenciales
* `docker ps`: Verificar contenedores activos.
* `docker-compose pause` / `docker-compose unpause`: Pausar y reanudar procesamiento sin destruir recursos.
* `docker-compose down`: Detener y eliminar la infraestructura completamente.

## Documentación

El sistema cuenta con documentación técnica exhaustiva para desarrolladores, con un índice centralizado en el directorio `docs/`:

1. [Arquitectura y Flujo de Datos](docs/architecture.md)
2. [Sistema de Detección](docs/detector.md)
3. [Configuración y Despliegue](docs/deployment.md)

## Estado y Evolución del Proyecto
**Estado Actual:** MVP Madurado y Optimizado para IoT (ARM64).
El código base es capaz de ejecutarse nativamente en sistemas SBC como la Raspberry Pi, integrándose de forma impecable con subsistemas de hardware modernos (libcamera) y administrando sus recursos térmicos mediante exportación especializada de modelos (NCNN).

**Líneas de Evolución Futuras:**
* Creación de un Dashboard web remoto para visualizar los eventos analíticos, reportes de salud del dispositivo y retransmisiones bajo demanda (Video on Demand).
* Implementación de notificaciones asíncronas push (Telegram, Email, Webhooks empresariales) en el `action_service`.
* Actualización para soportar aceleradores de hardware tensorial dedicados (Google Coral TPU / Hailo AI).
