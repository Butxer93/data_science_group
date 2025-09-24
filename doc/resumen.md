# Informe Técnico de Calidad
## Proyecto: Recomendador de Repostajes/Carga con Ahorro y Sostenibilidad

**Autor:** Equipo DS  
**Fecha:** 24-09-2025  
**Stack:** FastAPI, Python 3.10+, scikit-learn, LightGBM (LTR), pandas, numpy

---

## 1. Objetivo y alcance

Construir un sistema que, ante la intención *“quiero repostar/cargar”* o la planificación de una **ruta**, recomiende **dónde parar** optimizando:

- **Ahorro económico** (precio, desvío, tiempo de espera).
- **Sostenibilidad** (CO₂ por litro/kWh, prácticas de eco-conducción).
- **Escalabilidad** (API sencilla, modular, integrable en la plataforma de flota).

El sistema funciona con:
1) **Baseline explicable** (fórmula de *scoring* con pesos ajustables).  
2) **Learning-to-Rank (LTR)** con **LightGBM Ranker** (LambdaRank) y *fallback* a RandomForest.

---

## 2. Datos: esquema y diccionario

### 2.1 Vehículos
Campos clave (por vehículo):
- `vehicle_id` *(ID único)*  
- `plate` *(matrícula)*  
- `fuel_type ∈ {diesel, gasoline, ev, phev}`  
- `vehicle_type` *(car, van, truck, bus...)*  
- `empty_weight_kg`, `height_m`  
- **Combustión:** `avg_consumption_l_100km`, `tank_capacity_l`, `emissions_factor_g_per_l`  
- **EV/PHEV:** `battery_kwh`, `efficiency_kwh_100km`, `max_dc_kw`, `plug_types`  

### 2.2 Rutas
- `route_id`, `vehicle_id`  
- `start_lat`, `start_lon`, `end_lat`, `end_lon`  
- `via_points` (opcional), `avoid_tolls: bool`  
- Derivadas (enrutamiento): `route_polyline`, `route_distance_km`, `route_time_min`.
  

### 2.3 Estaciones de combustible
- `station_id`, `brand`, `lat`, `lon`, `fuel_type`  
- `price_per_liter`, `wait_min`  
- (Opcional) horarios, `price_updated_at`.

### 2.4 Puntos de carga EV
- `charge_id`, `name`, `lat`, `lon`  
- `network`, `plug_types`, `max_power_kw`  
- `price_eur_per_kwh`, `wait_min`, `availability` (si la fuente lo ofrece).

### 2.5 Telemetría (para hábitos)
- `vehicle_id`, `timestamp`, `lat`, `lon`, `speed`, `accel`, `idle_flag`, `odometer_km`.

**Fuentes externas (EV):**
- OpenChargeMap (API pública).
- datos.gob.es (catálogo español de recarga).

---

## 3. Ingeniería de características (features)

### 3.1 Nivel estación-candidato
- **Económicas**:  
  - `delta_price = price_area_mean - price_per_liter`  
  - `detour_km` (distancia al **corredor de ruta**; ver §3.4)  
  - `wait_min`  
  - `liters_needed` (combustión) o `kwh_needed` (EV)
- **Categorías**: `brand`, `fuel_type_station`, `vehicle_type`
- **Sostenibilidad**:
  - `co2_stop_kg = liters_needed * emissions_factor_g_per_l / 1000`
  - `shadow_price_co2_eur_per_kg` (parámetro de política)

### 3.2 Nivel vehículo
- `vehicle_type`, `fuel_type`, `empty_weight_kg`, `height_m`  
- **Rango combustión**:  
  \[
  range\_{km} = \frac{tank\_capacity\_l}{avg\_consumption\_{l/100km}} \times 100
  \]
- **Rango EV**:  
  \[
  range\_{km} = \frac{battery\_{kWh}}{efficiency\_{kWh/100km}} \times 100
  \]

### 3.3 Nivel ruta/consulta
- `distance_to_dest_km`, `route_pos_km`, `avoid_tolls`  
- `price_area_mean` (media de zona/ventana)  
- `weekday`, `hour` (disponibles para estacionalidad).

### 3.4 Cálculo de desvío (detour) al corredor de ruta
- Generamos una **polilínea** entre inicio/fin (stub de enrutamiento).  
- `detour_km = distance(point, polyline)` usando una proyección equirectangular local y distancia al **segmento**.  
- Se filtran candidatos con `detour_km ≤ R` (p. ej. 10 km).

---

## 4. Modelado

### 4.1 Baseline explicable (sin aprendizaje)
$$\textbf{score}_s = w_€ \cdot (\Delta \text{precio} \cdot \text{litros}) - w_{desv} \cdot (\text{km extra} \cdot €/km) - w_{t} \cdot (\beta \cdot \text{espera\_min}) - w_{CO2} \cdot (\text{CO₂}_{kg} \cdot \text{precio\_sombra})
$$

- Pesos configurables por **política**: *cost*, *balanced*, *sustainability*.
- Ventaja: **interpretabilidad** y robustez con pocos datos.

### 4.2 Learning-to-Rank (LTR)
- **Algoritmo**: LightGBM **LGBMRanker** (`objective="lambdarank"`).  
- **Entrada**: lista de candidatos por **consulta** (grupo).  
- **Etiqueta de relevancia**:  
  - Binaria (1 si fue elegida; 0 si no), o  
  - Multi-grado según **ahorro neto económico-ambiental**.  
- **Features** (§3.1–3.3).  
- **Métricas**: **NDCG@k**, **MAP**, **P@1**.  
- **Split**: por **consulta/ruta** (evita fuga).  
- **Hiperparámetros base**:  
  - `n_estimators=300`, `learning_rate=0.08`, `num_leaves=63`,  
  - `subsample=0.9`, `colsample_bytree=0.9`, `random_state=42`.

> **Por qué LambdaRank**: optimiza directamente métricas de ranking con **gradientes ponderados por posición** (los aciertos arriba pesan más).

### 4.3 Hábitos (módulo complementario)
- **Clasificación** (RandomForest / LogReg) de eco-eficiencia (label por percentil de `L/100km` **normalizado por segmento**).  
- **Clustering** (KMeans) para **segmentar estilos** y priorizar consejos.

---

## 5. Entrenamiento y validación

### 5.1 Preparación del set LTR
- Construcción de **grupos** = consultas (ruta, vehículo, ventana).  
- Candidatos = estaciones dentro del corredor y con **fuel compatible**.  
- `y ∈ {0,1}` (o multi-grado).  
- **Preprocesado**:  
  - `StandardScaler` numéricas, `OneHotEncoder` categóricas.  
  - `ColumnTransformer` + `Pipeline` (persistido con joblib).

### 5.2 Esquemas de validación
- **Hold-out por consulta** (25% test).  
- **GroupKFold** (CV por grupo).  
- *Opcional*: **validación temporal** si hay histórico con series.

### 5.3 Métricas
- **NDCG@k**:  
  $$
  NDCG@k = \frac{DCG@k}{IDCG@k},\;\;
  DCG@k = \sum_{i=1}^k \frac{2^{rel_i}-1}{\log_2(i+1)}
  $$
- **MAP**: media de precisión en cada *hit* sobre consultas.  
- **P@1**: precisión en la primera posición.  
- **Negocio**: €/km, CO₂/km, adopción de recomendación.

---

**Observaciones:**
- LTR mejora la **colocación top** frente al baseline.  
- RF capta señales no lineales pero no optimiza ranking directamente.

---

## 7. Sostenibilidad: modelado y reporte

- **CO₂ por parada** (combustión):  
  $$CO_{2}{kg} = \text{litros} \times \frac{emissions\_factor\_{g/L}}{1000}$$
  
- **Precio sombra** configurable: penaliza estaciones que, a igualdad de precio, generan mayor huella.  
- **EV**: se priorizan **potencia**, **precio €/kWh** y **compatibilidad de conector**; impacto CO₂ indirecto según mix eléctrico (si se modela).


---


