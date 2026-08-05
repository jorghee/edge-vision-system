# Sistema de Detección

El núcleo de la visión artificial reside en el módulo `detector`. Su responsabilidad principal es procesar las imágenes en crudo, identificar a las personas presentes en la escena y evaluar el cumplimiento del uso de Equipos de Protección Personal (EPP).

## Abstracción de Cámara

Para facilitar el despliegue tanto en entornos de desarrollo local como en sistemas IoT (como Raspberry Pi), se incluye el módulo `camera.py`. Este componente expone una interfaz común (`Camera`) soportada por distintos backends de captura:

* **OpenCVCamera**: Utiliza la librería estándar `cv2.VideoCapture`. Está diseñado para inicializar y capturar de cámaras USB (webcams) en arquitecturas x86 (Laptops o Servidores).
* **PiCamera**: Utiliza la librería `picamera2` e interactúa nativamente con `libcamera`. Es el método oficial y de alto rendimiento para interactuar con cámaras conectadas por CSI en Raspberry Pi (ej. Raspberry Pi Camera Rev 1.3).

La selección del backend se realiza dinámicamente mediante la variable de entorno `CAMERA_BACKEND` (`"opencv"`, `"picamera2"`, o `"auto"`).

## Detección Base (Personas)

El procesamiento inicial de cada frame se ejecuta con **YOLOv8** (usualmente la variante ultraligera `yolov8n`). La red neuronal base está pre-entrenada en el dataset COCO, lo que permite aprovechar su alta precisión. El script restringe la detección para aislar exclusivamente la clase `0` (Persona) e ignorar el resto del espectro del modelo.

## Detección de EPP (Cascos y Chalecos)

Una vez ubicadas las personas en la escena (Bounding Boxes), el detector recorta estas regiones específicas y las procesa para identificar equipo de seguridad.

Existen dos estrategias concurrentes para este fin:

1. **Modelo Fine-Tuned (Primario)**:
   Si el sistema detecta que se ha descargado un modelo especializado (ej. `ppe_detector.pt`), realiza la inferencia directamente sobre el recorte de la persona. Estos modelos proveen alta precisión para detectar si hay un casco presente de forma robusta frente a diferentes niveles de iluminación.

2. **Análisis por Color HSV (Fallback y Complemento)**:
   Si el modelo EPP primario no está disponible, o para detectar chalecos (para los cuales el modelo secundario no ha sido entrenado), se aplica un enfoque clásico de segmentación de color:
   * El recorte de la persona se divide geométricamente (ej. el tercio superior corresponde a la cabeza, el sector medio al torso).
   * La región resultante se transfiere del espacio de color BGR a HSV (Tono, Saturación, Valor).
   * Se evalúan proporciones de píxeles activos aplicando máscaras basadas en colores típicos de los EPP (Amarillo, Naranja, Rojo, Blanco).
   * Si la densidad del color deseado supera un umbral prestablecido en las áreas de interés, se dictamina como "detectado".

## Evaluador de Severidad

Dependiendo de la presencia del casco y el chaleco en la persona analizada, el sistema asigna una clasificación general al evento:
* **`none` (Evento `ppe_compliant`)**: Ambos elementos (casco y chaleco) fueron detectados correctamente.
* **`high` (Evento `no_helmet` o `no_vest`)**: Se identifica una falta parcial de los EPP.
* **`critical` (Evento `no_helmet_no_vest`)**: La persona en escena carece de los dos elementos obligatorios.

Además, cuando el procesador evalúa un frame sin ninguna persona detectada, emite un evento preventivo `clear`, informando al sistema que la zona visual se encuentra despejada.

## Monitoreo de Hardware (Health Monitor)

Procesar inferencia continua genera estrés sostenido en los procesadores locales. El módulo `health_monitor.py` funciona como un script satélite que extrae métricas de telemetría del hardware.

Cada 30 segundos, publica la siguiente información en el topic `edge/health`:
* Temperatura del CPU (°C).
* Porcentaje de uso del procesador (basado en _load average_).
* Estado de la memoria RAM (Total, Usada y Porcentaje).
* Uso de disco de la partición raíz.
* Estado de Throttling del SoC (particularmente importante para evitar daños en Raspberry Pi).

Esta información adicional permite ampliar las reglas de eKuiper para prevenir sobrecalentamientos térmicos y administrar la disponibilidad del dispositivo IoT de forma autónoma.
