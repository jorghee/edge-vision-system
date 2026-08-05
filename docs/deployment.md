# Configuración y Despliegue

Este documento abarca los diferentes enfoques de instalación y ejecución del Edge Vision System. El proyecto ofrece dos vías principales de ejecución: un modelo totalmente en contenedores (ideal para pruebas y desarrollo en PCs), y un modelo híbrido para dispositivos IoT (ej. Raspberry Pi) donde las restricciones de hardware demandan ejecuciones nativas para maximizar la compatibilidad.

## Requisitos y Dependencias Previas

Independientemente de la estrategia elegida, se debe contar con:
* **Docker y Docker Compose**: Instalado para orquestar la infraestructura subyacente.
* **Conexión MQTT**: Mosquitto habilitado operando convencionalmente en el puerto `1883`.
* **eKuiper API**: El motor de streaming en puerto `9081`.

---

## 1. Despliegue en Laptop o Servidor Local (x86_64)

Este método despliega todos los componentes, incluido el detector, mediante contenedores, lo cual garantiza paridad con entornos de desarrollo y evita el enredo de dependencias de sistema.

### Levantar los Contenedores

Ejecuta el siguiente comando en la raíz del proyecto para construir imágenes locales y levantar todos los servicios descritos en `docker-compose.yml`:
```bash
docker-compose up --build
```
> [!NOTE]
> En entornos Linux, la cámara USB conectada se mapeará automáticamente asumiendo `/dev/video2`. Ajusta este parámetro en el YAML según tu montaje de hardware (ej. `/dev/video0`).

### Configurar Reglas en eKuiper

Una vez que el motor esté operando, se deben inicializar los "Streams" y asignar las reglas. Los comandos concretos (mediante `curl`) se detallan exhaustivamente en el archivo `README.md` de la raíz del proyecto. Estas reglas habilitarán el reenvío de alertas al topic de `edge/alerts` escuchado por el `action_service`.

---

## 2. Despliegue en Dispositivos IoT (Raspberry Pi)

Los despliegues en el borde de la red (Edge Computing real) afrontan limitaciones serias de computación. Este proyecto está diseñado para funcionar en hardware **Raspberry Pi (OS Bookworm 64-bit)** aprovechando las cámaras nativas CSI.

### Consideraciones Críticas de Arquitectura (IoT)

* **Compatibilidad de Cámara**: Las cámaras CSI modernas operan utilizando `libcamera` y el puente `picamera2`. No se exponen con facilidad como dispositivos Linux clásicos (`/dev/videoX`) para interactuar con Docker sin un proceso intrincado de paso de dispositivos.
* **Modelo Híbrido**: Se recomienda levantar la infraestructura (MQTT, eKuiper y Action Service) usando contenedores, pero ejecutar el script Python del **Detector directamente de forma nativa**.
* **Limitación de Inferencia de CPU (PyTorch)**: Ejecutar el modelo original de PyTorch (`.pt`) ralentiza la ejecución, pudiendo tomar varios segundos por cada frame procesado en la placa.

### Paso a Paso de Instalación

1. **Arrancar la Infraestructura en Contenedores**:
   En la RPi, utiliza el archivo Compose adaptado específicamente para el hardware ARM, el cual no incluye al detector:
   ```bash
   docker compose -f docker-compose.rpi.yml up -d
   ```

2. **Preparar el Entorno Nativo para el Detector**:
   Crea el entorno virtual y agrega las dependencias necesarias de IoT, que incluyen compilados matemáticos y soporte a la cámara:
   ```bash
   cd detector
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements-rpi.txt
   ```

3. **Optimización de Modelo (Exportación a NCNN)**:
   Las inferencias a tiempo real en ARM se benefician del uso del motor **NCNN**, capaz de ejecutar operaciones optimizadas con el repertorio SIMD NEON. 
   **En tu equipo principal (laptop):**
   ```bash
   python export_model.py --base models/yolov8n.pt --format ncnn
   ```
   A continuación, copia la carpeta procesada `yolov8n_ncnn_model/` directamente a la RPi bajo el directorio `detector/models/`.

4. **Iniciando el Detector (Raspberry Pi)**:
   Ejecuta el asistente Bash del detector en el entorno IoT, el cual auto-seleccionará el motor de red neuronal `NCNN`, inicializará el backend de Picamera2 y se sincronizará con la red local de Docker Compose previamente creada:
   ```bash
   ./run_rpi.sh
   ```

Al implementar esta división, el procesamiento visual de IA extrae todo el rendimiento posible del dispositivo y las reglas de integración quedan enclaustradas dentro de las herramientas estandarizadas de IoT en contenedores.
