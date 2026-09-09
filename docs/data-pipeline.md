# Pipeline de Datos y Observabilidad

Una vez que los datos han sido extraídos y filtrados inteligentemente en el dispositivo Edge, el desafío consiste en transportarlos de forma segura, estandarizarlos y almacenarlos eficientemente para el análisis a largo plazo en el Servidor Central.

El sistema implementa una tubería de datos (Data Pipeline) unificada, que utiliza el mismo bus de transporte tanto para los eventos de negocio (Detecciones EPP) como para la telemetría operativa (Salud del Hardware).

## Transporte: MQTT

MQTT (Message Queuing Telemetry Transport) es el estándar de facto para redes IoT. Se utiliza el broker **Mosquitto** desplegado en el servidor central.

¿Por qué MQTT en lugar de HTTP REST?
1. **Ligereza:** El overhead de los headers MQTT es de apenas 2 bytes, comparado con cientos de bytes en HTTP.
2. **Asincronía y Resiliencia:** Si el servidor central cae, los clientes MQTT en el borde intentarán reconectarse automáticamente. Si se configura QoS (Quality of Service) adecuado, los mensajes generados durante la desconexión se encolan y se envían en ráfaga una vez restaurado el enlace.
3. **Patrón Pub/Sub:** Permite escalar fácilmente. Un dispositivo edge simplemente "pública" su alerta; no necesita conocer la dirección IP de la base de datos final. Telegraf se "suscribe" a ese tópico y se encarga del ruteo.

### Taxonomía de Tópicos

El diseño del espacio de nombres (Topic Namespace) permite aislar los tipos de eventos:

- `edge/alerts`: Alertas de alta severidad (`critical`, `high`). Útil para disparar sirenas, enviar notificaciones SMS o correos inmediatos (eventualmente consumidos por un Action Service externo).
- `edge/monitor`: Todos los eventos (incluso los de severidad leve). Se utiliza exclusivamente para poblar los dashboards analíticos.
- `edge/metrics/{camera_id}`: Telemetría de hardware generada por el Node Agent de cada dispositivo específico.

## Traducción e Ingesta: Telegraf

[Telegraf](https://www.influxdata.com/time-series-platform/telegraf/) actúa como el agente recolector y traductor universal.

1. **Suscripción:** Está configurado con el plugin `mqtt_consumer` escuchando los tópicos mencionados.
2. **Parseo JSON (Data Normalization):** Recibe los payloads en crudo de eKuiper (JSON) y los desarma. Identifica qué campos son `Tags` (metadatos indexados, ej. `camera_id`, `severity`) y qué campos son `Fields` (valores métricos, ej. `confidence`, `cpu_percent`).
3. **Escritura por lotes (Batching):** No inserta evento por evento en la base de datos (lo cual saturaría el disco). Agrupa los eventos durante 10 segundos o hasta acumular un lote grande, y realiza una única inserción HTTP a InfluxDB optimizando el I/O del disco.

## Almacenamiento: InfluxDB

[InfluxDB](https://www.influxdata.com/) es una base de datos Time-Series (TSDB). A diferencia de las bases de datos relacionales tradicionales (PostgreSQL) o de documentos (MongoDB), las TSDB están optimizadas de fábrica para manejar miles de inserciones por segundo, siempre asociadas a una marca temporal (`timestamp`).

InfluxDB permite la agregación nativa a alta velocidad. Por ejemplo, calcular "el promedio móvil de la temperatura de la cámara 3 en las últimas 4 semanas con saltos de 1 hora" toma fracciones de segundo.

En InfluxDB se mantienen dos conjuntos de datos (`measurements`):
- `ppe_events`: Datos analíticos del comportamiento de los trabajadores.
- `device_metrics`: Registro histórico de la salud térmica y de red de las Raspberry Pi.

## Visualización y Monitoreo: Grafana

El servidor central expone **Grafana** como interfaz unificada de análisis. 

Grafana no está configurado para mostrar gráficos bonitos sin sentido; está estructurado como una herramienta de operaciones (NOC) para resolver problemas reales, separado en dos dimensiones operativas:

### 1. Dashboard: PPE Detection (Alertas y Predicciones)
Diseñado para el **Analista de Seguridad / Prevencionista de Riesgos**.
- Responde a preguntas de negocio: *¿Cuántos trabajadores han incumplido la normativa hoy? ¿Qué tipo de infracción es más común (sin casco vs sin chaleco)? ¿A qué hora ocurren más incidentes?*
- Cuenta con una tabla de **Eventos Críticos** que renderiza el `snapshot` (Base64) adjunto a los eventos severos, permitiendo auditar visualmente el error humano sin tener que ver el video crudo.

### 2. Dashboard: Device Health (Observabilidad)
Diseñado para el **Ingeniero de Infraestructura / DevOps**.
- Responde a preguntas operativas: *¿Alguna cámara perdió conexión WiFi recientemente? ¿Están las Raspberry Pi superando los 80°C y sufriendo estrangulamiento térmico? ¿Cuántos frames por minuto están procesando realmente? ¿El modelo de IA está lanzando excepciones?*
- Este dashboard permite predecir fallos de hardware o red antes de que resulten en pérdida de datos.
