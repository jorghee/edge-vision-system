# Edge Vision System

Bienvenido a la documentación oficial del **Edge Vision System**. Este proyecto implementa una solución de visión artificial ligera orientada al Edge Computing (procesamiento en el borde) para la detección de personas y el cumplimiento del uso de Equipos de Protección Personal (EPP), específicamente cascos y chalecos.

## Resumen del Proyecto

El sistema está diseñado para capturar video en tiempo real desde una cámara local, procesar los frames para identificar personas y verificar el uso de EPP, y publicar eventos estructurados. Utiliza un motor de reglas ligero (eKuiper) para filtrar el ruido y reenviar únicamente las alertas relevantes (ej. violaciones de seguridad) hacia un servicio de acción.

## Tabla de Contenidos

1. [Arquitectura y Flujo de Datos](architecture.md)
2. [Sistema de Detección](detector.md)
3. [Configuración y Despliegue](deployment.md)

## Componentes Principales

El sistema se compone de cuatro piezas fundamentales, orquestadas y comunicadas a través de MQTT:

* **Detector de Visión**: Componente en Python que captura el flujo de video, ejecuta inferencia utilizando YOLOv8 y publica eventos detallados sobre cada frame procesado.
* **Broker MQTT (Mosquitto)**: Actúa como el bus central de mensajería del sistema, permitiendo una comunicación asíncrona y desacoplada entre todos los servicios.
* **Motor de Reglas (eKuiper)**: Un motor SQL ligero para Edge Computing. Se encarga de procesar el flujo continuo de eventos del detector, aplicar lógica de filtrado por severidad y redirigir las alertas críticas.
* **Action Service**: Servicio responsable de escuchar las alertas filtradas por eKuiper y ejecutar las acciones de respuesta correspondientes (actualmente logging y publicación de recomendaciones).

## Dependencias Clave

* **Docker & Docker Compose**: Para orquestar los servicios de infraestructura.
* **Python 3.11+**: Lenguaje principal de los servicios personalizados (detector y action_service).
* **Ultralytics (YOLOv8)**: Framework para detección de objetos.
* **OpenCV / Picamera2**: Manejo de captura de imágenes y manipulación matricial.
* **Eclipse Mosquitto**: Broker MQTT.
* **LF Edge eKuiper**: Motor de streaming y procesamiento de reglas en el borde.
