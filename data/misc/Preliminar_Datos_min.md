1. Viajes / Telemetría

| Campo         | Descripción                                                                  |
|---------------|------------------------------------------------------------------------------|
| vehicle_id    | Identificador único del vehículo                                             |
| timestamp     | Fecha y hora de la lectura                                                   |
| lat, lon      | Posición geográfica (coordenadas GPS)                                        |
| odometer_km   | Lectura del odómetro en kilómetros                                           |

👉 **Uso:** Medir eficiencia (L/100 km, CO₂/km), hábitos de conducción (ralentí, aceleraciones bruscas), patrones de movilidad.

2. Repostajes
| Campo           | Descripción                                         |
|-----------------|-----------------------------------------------------|
| vehicle_id      | Vehículo que realiza el repostaje                   |
| station_id      | Estación de servicio utilizada                      |
| datetime        | Fecha y hora del repostaje                          |
| liters          | Litros a cargar                                     |


👉 **Uso:** Cálculo de coste/km, ranking de estaciones, predicción de repostajes futuros.

3. Estaciones

| Campo             | Descripción                                                |
|-------------------|------------------------------------------------------------|
| station_id        | Identificador único de la estación                         |
| brand             | Marca de la estación (ej. Repsol, Shell)                   |
| lat, lon          | Ubicación geográfica                                       |

👉 **Uso:** Ranking de estaciones, recomendaciones de ahorro, optimización de rutas.


🚙 4. Vehículos

| Campo                    | Descripción                                                   |
|--------------------------|---------------------------------------------------------------|
| vehicle_id               | Identificador único del vehículo                              |
| segment                  | Tipo de vehículo (turismo, furgoneta, camión, flota etc.)     |
| model_vehicle            | modelo del vehículo                                           |

👉 **Uso:** Normalizar métricas por segmento, calcular emisiones, personalizar recomendaciones.

