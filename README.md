# Edge Vision System

Este proyecto implementa un sistema de visión inteligente. Está compuesto por dos servicios principales:

- **Detector**: Procesa imágenes o video para detectar personas y verificar que utilizan el equipo de protección personal (EPP).
- **Action Service**: Gestiona las alertas relacionadas con el sistema de detección.

## Requisitos previos

Se necesita tener instalados en tu sistema las siguientes herramientas:

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

## Instalación y Ejecución

1. Clona este repositorio en tu sistema local:

    ```bash
    git clone <URL-del-repositorio>
    cd edge-vision-system
    ```

2. Inicia los servicios utilizando Docker Compose:

    ```bash
    docker-compose up --build
    ```

   Esto construirá las imágenes de Docker y levantará los servicios definidos en el archivo `docker-compose.yml`.

3. Para detener los servicios, ejecuta:

    ```bash
    docker-compose down
    ```

## Servicios

### Detector

- **Ruta de construcción**: `detector/`
- **Dependencias**:
  - Python 3.11
  - Librerías especificadas en `requirements.txt`
- **Variables de entorno**:
  - `MQTT_BROKER`: Dirección del broker MQTT (por defecto: `mqtt`)
  - `MQTT_PORT`: Puerto del broker MQTT (por defecto: `1883`)
  - `CAMERA_INDEX`: Índice del dispositivo de cámara (por defecto: `2`)
  - ... y más variables configuradas en `docker-compose.yml`.

### Action Service

- **Ruta de construcción**: `action_service/`
- **Dependencias**:
  - Python 3.11
  - Librerías especificadas en `requirements.txt`
- **Variables de entorno**:
  - `MQTT_BROKER`: Dirección del broker MQTT (por defecto: `mqtt`)
  - `ALERT_TOPIC`: Tópico MQTT para alertas (por defecto: `edge/alerts`)

---

## Comandos útiles

- Para ver los contenedores en ejecución:
  ```bash
  docker ps
  ```

- Para acceder a un contenedor en ejecución:
  ```bash
  docker exec -it <nombre-del-contenedor> bash
  ```

- Para limpiar imágenes y contenedores antiguos:
  ```bash
  docker system prune
  ```

## Estructura del proyecto

- `docker-compose.yml`: Define y configura los servicios del sistema.
- `detector/`: Contiene el servicio de detección basado en YOLO.
- `action_service/`: Contiene el servicio de gestión de alertas.
