# Procesamiento Edge: eKuiper y Observabilidad Local

El verdadero núcleo de ingeniería de este sistema reside en el dispositivo Edge (la Raspberry Pi). El objetivo principal aquí es procesar el inmenso flujo de datos de video lo más cerca posible de la fuente para extraer únicamente el valor de negocio.

## El Motor de Reglas: LF Edge eKuiper

[eKuiper](https://ekuiper.org/) es un motor de análisis de datos y procesamiento de flujos ligero, diseñado específicamente para dispositivos con recursos limitados en entornos IoT. 

En lugar de construir un script monolítico en Python que maneje la cámara, la inferencia de IA, la lógica de negocio, y la conexión de red (lo cual es propenso a caídas, fugas de memoria y difícil de actualizar), utilizamos eKuiper para orquestar todo el pipeline de forma declarativa.

### Pipeline de eKuiper

El flujo de procesamiento dentro de eKuiper se define mediante Streams (fuentes de datos) y Rules (lógica SQL y destinos).

1.  **Source (Captura):** Utilizamos el plugin nativo de eKuiper para conectarnos al servidor RTSP local (provisto por MediaMTX). Extrae frames crudos a un intervalo controlado (ej. 1 frame por segundo).
2.  **Inferencia (Portable Plugin):** eKuiper permite inyectar código Python personalizado mediante su sistema de **Portable Plugins**. Nuestro plugin `ppeInference` recibe la matriz de bytes de la imagen y ejecuta YOLOv8, devolviendo un JSON estructurado con la severidad de la infracción y el nivel de confianza.
3.  **Filtrado SQL:** Las reglas de eKuiper aplican sentencias SQL estándar sobre el flujo infinito de detecciones JSON.
4.  **Sink (Destino):** Los resultados que pasan los filtros SQL son enviados al servidor central mediante un sumidero (Sink) MQTT.

### Reglas SQL y Reducción de Ruido

La capacidad de eKuiper para aplicar SQL sobre flujos de datos en tiempo real es lo que permite reducir drásticamente el consumo de red.

Por ejemplo, la regla `ppe_alert_critical` se define de la siguiente manera:

```sql
SELECT 
    camera_id, 
    event_type, 
    severity, 
    confidence, 
    persons_detected, 
    snapshot, 
    timestamp
FROM camera_frames
WHERE severity = 'critical'
```

Esta consulta opera como una compuerta: si la cámara enfoca una escena vacía durante 8 horas (generando ~28,800 frames), la regla SQL filtrará todo. **Cero bytes serán transmitidos por la red hacia el servidor central.** Solo cuando `severity = 'critical'` (ej. un trabajador sin casco ni chaleco) se abrirá la compuerta y el JSON resultante será empujado a MQTT.

## Modelo de IA (Caso de Uso EPP)

El plugin de inferencia encapsula un pipeline de visión artificial:

1.  **Detección de Personas (YOLOv8):** Utiliza un modelo YOLO ultraligero exportado a formato TFLite (TensorFlow Lite) o NCNN, optimizado específicamente para los procesadores ARM (NEON SIMD) de la Raspberry Pi.
2.  **Análisis por Instancia:** Por cada persona detectada, se recorta su imagen (bounding box) de forma individual.
3.  **Análisis de Casco:** Se evalúa la parte superior del recorte buscando geometría y colores (HSV) correspondientes a cascos de seguridad industrial.
4.  **Análisis de Chaleco:** Se evalúa la región del torso buscando patrones y colores de alta visibilidad (amarillo/naranja neón) mediante segmentación.
5.  **Generación de Snapshots:** Si se confirma una infracción grave, el plugin codifica el frame infractor en Base64 para adjuntarlo como evidencia visual (`snapshot`) en el mensaje JSON.

> [!NOTE]
> La arquitectura del plugin permite que el modelo de IA sea intercambiable. Podría sustituirse el modelo de EPP por uno de lectura de matrículas (ALPR) o detección de fuego, sin tener que alterar el resto de la infraestructura (MQTT, Telegraf, InfluxDB).

## Observabilidad del Dispositivo (Node Agent)

Monitorear el rendimiento de un dispositivo IA en el borde es crítico. La Raspberry Pi operando modelos de Deep Learning puede sufrir sobrecalentamiento rápido, lo que induce al estrangulamiento térmico (Thermal Throttling) y degrada severamente el rendimiento.

Para esto, se diseñó el **Node Agent**:

*   Es un script Python ultra-ligero (`< 20 MB` de RAM) que se ejecuta como servicio secundario.
*   Recolecta telemetría profunda cada 30 segundos:
    *   Uso de CPU y RAM.
    *   Temperatura del SoC (`vcgencmd`).
    *   Métricas de disco y red.
    *   Estado de salud del motor eKuiper (consultando su API REST interna para extraer latencias de inferencia y conteo de errores).
*   Publica estos datos en el tópico `edge/metrics/#`.

Esta telemetría permite que el dashboard de **Device Health** en el servidor central alerte inmediatamente si un nodo en el terreno está a punto de fallar por sobrecalentamiento o falta de memoria.
