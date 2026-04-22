# Edge Vision System

## Comandos Docker Esenciales

Aquí se detallan los comandos clave para gestionar este proyecto utilizando Docker y Docker Compose.

### Iniciar el Proyecto
Para construir las imágenes y ejecutar los servicios:
```bash
docker-compose up --build
```
Esto compila las imágenes y lanza los contenedores según `docker-compose.yml`.

### Verificar Servicios en Estado Activo
```bash
docker ps
```

### Pausar Servicios
```bash
docker-compose pause
```
Esto detiene la ejecución de los contenedores temporalmente.

### Reanudar Servicios Pausados
```bash
docker-compose unpause
```

### Detener Servicios
Para detener y eliminar contenedores en ejecución:
```bash
docker-compose down
```

### Reiniciar Servicios
Si deseas reiniciar todos los servicios (no requiere rebuilding de imágenes):
```bash
docker-compose restart
```

### Limpiar Recursos No Utilizados
Para limpiar recursos no utilizados (contenedores, imágenes, volumenes):
```bash
docker system prune
```

## Ekuiper rules

Primero limpiar

```bash
# Eliminar reglas y stream anteriores
curl -X DELETE http://localhost:9081/rules/alert_high_severity
curl -X DELETE http://localhost:9081/rules/alert_ppe_violation
curl -X DELETE http://localhost:9081/streams/camera_events

# Verificar que quedó limpio
curl -s http://localhost:9081/rules | python3 -m json.tool
curl -s http://localhost:9081/streams | python3 -m json.tool
# Ambos deben retornar: []

``` 

Crear el stream

```bash
curl -X POST http://localhost:9081/streams \
  -H "Content-Type: application/json" \
  -d '{
    "sql": "CREATE STREAM camera_events() WITH (DATASOURCE=\"camera/events\", FORMAT=\"json\", TYPE=\"mqtt\", CONF_KEY=\"default\")"
  }'

# Verificar
curl -s http://localhost:9081/streams | python3 -m json.tool
# Esperado: ["camera_events"]

```

Crear las reglas

```bash
curl -X POST http://localhost:9081/rules \
  -H "Content-Type: application/json" \
  -d '{
    "id": "alert_critical",
    "sql": "SELECT camera_id, event_type, severity, confidence, timestamp, metadata FROM camera_events WHERE severity = '\''critical'\''",
    "actions": [
      {
        "mqtt": {
          "server": "tcp://mqtt:1883",
          "topic": "edge/alerts",
          "qos": 1
        }
      },
      { "log": {} }
    ]
  }'

```

```bash
curl -X POST http://localhost:9081/rules \
  -H "Content-Type: application/json" \
  -d '{
    "id": "alert_high",
    "sql": "SELECT camera_id, event_type, severity, confidence, timestamp, metadata FROM camera_events WHERE severity = '\''high'\''",
    "actions": [
      {
        "mqtt": {
          "server": "tcp://mqtt:1883",
          "topic": "edge/alerts",
          "qos": 1
        }
      },
      { "log": {} }
    ]
  }'

```

```bash
curl -X POST http://localhost:9081/rules \
  -H "Content-Type: application/json" \
  -d '{
    "id": "monitor_all",
    "sql": "SELECT camera_id, event_type, severity, confidence, timestamp FROM camera_events WHERE event_type != '\''clear'\''",
    "actions": [
      {
        "mqtt": {
          "server": "tcp://mqtt:1883",
          "topic": "edge/monitor",
          "qos": 0
        }
      }
    ]
  }'

```
