# Edge Vision System: Plataforma de Datos IoT & Edge Computing

Este proyecto es una **Infraestructura de Ingeniería de Datos para entornos Edge e IoT**, diseñada para gestionar, procesar y transportar eficientemente grandes volúmenes de datos generados en el borde de la red. 

Como **caso de uso demostrativo**, el sistema implementa un modelo de Inteligencia Artificial (YOLOv8) para la **Detección de Equipos de Protección Personal (EPP)**, evaluando en tiempo real si los trabajadores portan casco y chaleco de seguridad. 

Sin embargo, el núcleo tecnológico del proyecto no es el modelo de IA en sí, sino la **arquitectura distribuida** que permite que este procesamiento ocurra de forma eficiente en dispositivos de bajos recursos (ej. Raspberry Pi 4) sin saturar el ancho de banda hacia un servidor central.

## Propósito Fundamental

En implementaciones industriales reales (minería, manufactura, construcción), enviar secuencias de video continuo a la nube para su procesamiento es insostenible debido a los costos de red, la latencia y la intermitencia de la conexión.

Este proyecto resuelve ese desafío implementando un paradigma **Edge-First**:

1. **Procesamiento Distribuido Local:** Los datos pesados (video) son adquiridos, analizados y descartados localmente dentro de cada dispositivo Edge.
2. **Filtrado Inteligente:** Solo los eventos estructurados (alertas y métricas en formato JSON ligero) se envían a través de la red, reduciendo el consumo de ancho de banda en más de un 99%.
3. **Transporte Resiliente:** Se utiliza mensajería MQTT para enviar los eventos estructurados de forma confiable al servidor central.
4. **Observabilidad Centralizada:** Todos los eventos y métricas de salud (CPU, RAM, temperatura) de los múltiples dispositivos se consolidan en InfluxDB y se analizan dinámicamente mediante Grafana.

## Arquitectura de Alto Nivel

El sistema se divide físicamente en dos componentes principales: la red de dispositivos perimetrales (Edge Devices) y el cerebro centralizador (Central Server).

```mermaid
graph LR
    subgraph "Edge Network (Múltiples Dispositivos)"
        ED1[Raspberry Pi 4<br/>(Edge Device 1)]
        ED2[Dispositivo N<br/>(Edge Device N)]
    end
    
    subgraph "Central Server (Local o Nube)"
        MQTT[Mosquitto<br/>Broker MQTT]
        TLG[Telegraf<br/>Data Collector]
        DB[(InfluxDB<br/>Time-Series)]
        GF[Grafana<br/>Dashboards]
    end

    ED1 -- "Eventos JSON (MQTT)" --> MQTT
    ED2 -- "Eventos JSON (MQTT)" --> MQTT
    
    MQTT --> TLG
    TLG --> DB
    DB --> GF
```

> [!NOTE]
> Para una explicación técnica detallada del flujo interno, consulta [Arquitectura del Sistema](docs/architecture.md).

## Casos de Uso y Aplicabilidad

Aunque la implementación actual está configurada para el monitoreo de **Equipos de Protección Personal (EPP)**, la infraestructura de datos es agnóstica. Este patrón arquitectónico (Streaming Local → IA en el Borde → Reglas SQL → MQTT → Time-Series DB) puede extrapolarse a sectores intensivos en datos:

- **Sector Minero:** Monitoreo de vibraciones de maquinaria pesada, control de fatiga en conductores, detección perimetral en áreas sin cobertura 5G/LTE estable.
- **Sector Bancario y Retail:** Procesamiento distribuido de eventos de flujo de clientes en sucursales, observabilidad de infraestructura local, y alertas de seguridad, evitando enviar cientos de streams RTSP hacia el datacenter central.

## Documentación Técnica

La documentación ha sido estructurada exhaustivamente para cubrir el ciclo de vida completo de los datos. Recomendamos seguir este orden de lectura:

| Tema | Descripción | Enlace |
| :--- | :--- | :--- |
| **1. Arquitectura** | Topología de red, componentes, responsabilidades y diseño. | [architecture.md](docs/architecture.md) |
| **2. Procesamiento Edge** | El rol fundamental de **eKuiper**, el pipeline de IA y el filtrado local. | [edge-processing.md](docs/edge-processing.md) |
| **3. Pipeline de Datos** | Tubería completa: Mosquitto → Telegraf → InfluxDB → Grafana. | [data-pipeline.md](docs/data-pipeline.md) |
| **4. Despliegue** | Instrucciones parametrizables para Server y Dispositivos Edge. | [deployment.md](docs/deployment.md) |

## Tecnologías Principales

| Capa | Tecnología | Función Principal |
| :--- | :--- | :--- |
| **Streaming Edge** | [MediaMTX](https://github.com/bluenviron/mediamtx) | Adquisición y ruteo RTSP/WebRTC del flujo de cámara en el dispositivo. |
| **Procesamiento Edge**| [LF Edge eKuiper](https://ekuiper.org/) | Motor de procesamiento de flujo de datos, ejecución de IA, evaluación de reglas SQL. |
| **Modelos IA** | YOLOv8 (Ultralytics) | Detección de personas optimizada para dispositivos ARM (TFLite/NCNN). |
| **Transporte** | [Eclipse Mosquitto](https://mosquitto.org/) | Broker MQTT que actúa como columna vertebral de comunicación asíncrona. |
| **Almacenamiento** | [InfluxDB 2.x](https://www.influxdata.com/) | Base de datos de series temporales para almacenar eventos de IA y telemetría de hardware. |
| **Visualización** | [Grafana](https://grafana.com/) | Dashboards de monitoreo analítico (Alertas PPE y Observabilidad Edge). |
| **Orquestación** | Docker Compose | Contenerización y despliegue agnóstico al sistema operativo subyacente. |
