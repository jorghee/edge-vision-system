# Edge Vision System

Sistema de visión artificial y procesamiento de eventos diseñado para entornos de Edge Computing y hardware de recursos limitados (IoT).

## Objetivo y Propuesta de Valor
El sistema procesa flujos de video localmente en el dispositivo perimetral (ej. Raspberry Pi) utilizando algoritmos de visión artificial (YOLOv8) para detectar la presencia de personal y el cumplimiento del uso de Equipos de Protección Personal (EPP). 

La principal propuesta de valor es la eliminación de la transmisión constante de video hacia la nube. Mediante un filtrado inteligente perimetral, el sistema transmite únicamente alertas estructuradas en formato JSON ante eventos críticos, minimizando el consumo de ancho de banda y garantizando respuestas con latencias en el orden de milisegundos.

## Arquitectura General
El sistema implementa una arquitectura distribuida basada en microservicios comunicados asíncronamente mediante MQTT.

| Componente | Función Principal | Tecnología Clave |
| :--- | :--- | :--- |
| **Detector de Visión** | Captura de video, inferencia de red neuronal, evaluación EPP y telemetría. | Python, YOLOv8, NCNN, libcamera |
| **Broker de Mensajería** | Bus asíncrono de baja latencia para interconexión de componentes. | Eclipse Mosquitto |
| **Motor de Reglas** | Procesamiento de flujos de datos (*Stream Processing*) y filtrado SQL. | LF Edge eKuiper |
| **Action Service** | Servicio reactivo para integración externa y respuestas operativas. | Python, Paho MQTT |

> [!NOTE]
> La arquitectura completa y el flujo de datos están documentados exhaustivamente en [docs/architecture.md](docs/architecture.md).

## Despliegue en Hardware Real (Raspberry Pi)
El proyecto está optimizado para su ejecución en entornos físicos como la Raspberry Pi (OS Bookworm 64-bit) integrando cámaras nativas CSI. La estrategia de despliegue es híbrida para sortear las limitaciones de hardware: la infraestructura base opera en contenedores Docker, mientras que el módulo de inferencia corre de manera nativa sobre el sistema host.

### 1. Inicialización de Infraestructura (Docker)
Inicia los servicios de red, enrutamiento y procesamiento de reglas compilados nativamente para ARM64:
```bash
docker compose -f docker-compose.rpi.yml up -d
```

### 2. Configuración Automatizada de Reglas (eKuiper)
Ejecuta el script de aprovisionamiento para inicializar los *streams* de datos y los filtros SQL que gobiernan el pase de alertas críticas:
```bash
./setup_ekuiper.sh
```

### 3. Preparación de Entorno Nativo e Inferencia
Para asegurar el acceso directo de hardware a la cámara CSI (vía Picamera2) y el máximo rendimiento de inferencia (vía NCNN), inicializa el entorno de Python:
```bash
cd detector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-rpi.txt
```

### 4. Ejecución del Detector
Inicia el sistema de visión artificial. El script autodetectará el backend fotográfico adecuado y el modelo neuronal óptimo (NCNN vs PyTorch):
```bash
./run_rpi.sh
```

> [!TIP]
> Los detalles algorítmicos de la red (modelo primario y *fallback* colorimétrico) se encuentran en [docs/detector.md](docs/detector.md). Las instrucciones profundas de despliegue, incluyendo la conversión de modelos, están en [docs/deployment.md](docs/deployment.md).

## Características Implementadas

| Capacidad | Descripción Técnica |
| :--- | :--- |
| **Identificación Espacial** | Detección de personas en tiempo real usando el modelo YOLOv8. |
| **Análisis de EPP** | Verificación cruzada (modelo *fine-tuned* + *fallback* HSV) para detección de casco y chaleco. |
| **Abstracción de Captura** | Patrón *Factory* soportando *backends* de captura estándar USB (OpenCV) y CSI embebido (Picamera2). |
| **Monitoreo Térmico (IoT)** | Telemetría integrada para CPU, RAM y eventos de *throttling* térmico, esencial en SBCs expuestos. |
| **Aceleración SIMD** | Exportación de modelos PyTorch a NCNN, explotando eficientemente las instrucciones NEON en chips ARM. |

## Aplicaciones en la Industria
La naturaleza descentralizada del sistema permite su integración en proyectos de automatización pesada:
* **HSE Automatizado:** Restricción de acceso en zonas de riesgo basada en la validación perimetral de equipamiento, procesado in-situ.
* **Supervisión de Activos:** Señalización de alertas de proximidad a PLCs o balizas locales frente a detección humana en puntos ciegos de maquinaria móvil.
* **Redes Descentralizadas:** Despliegue masivo de nodos autónomos sin saturar redes operacionales IT/OT corporativas, centralizando solo metadatos críticos en tableros de control (SCADA).

## Entorno de Desarrollo y Pruebas (x86_64)
Para desarrollo inicial, pruebas de integración o emulaciones sin hardware IoT, el proyecto dispone de una versión integral contenerizada operando sobre cámaras web convencionales:
```bash
docker-compose up --build -d
./setup_ekuiper.sh
```

## Documentación Técnica
Todos los lineamientos de diseño, diagramas e implementaciones técnicas están documentados en `docs/`:
1. [Arquitectura y Flujo de Datos](docs/architecture.md)
2. [Sistema de Detección](docs/detector.md)
3. [Configuración y Despliegue](docs/deployment.md)
