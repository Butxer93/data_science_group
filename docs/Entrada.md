# App de recomendaciones de ahorro y escalabilidad en una plataforma de gestión de clientes/flota que entregua:

1. **Ranking de repostajes** *(estaciones accesibles ahora)*

2. **Consejos de Hábitos eficientes** (eco-driving)
  … considerando estándares de sostenibilidad *(coste + emisiones)*.

## Qué hace la app (end-to-end)

### Entrada

**Intención:** *“Deseo de repostar ahora”*.

**Contexto:** *vehicle_id*, *posición actual (lat/lon)*, *siguiente punto de ruta*, *nivel de tanque (%)*, *tipo de combustible*.

**Catálogo:** *estaciones cercanas* *(lat/lon, precio, espera, marca, fuel_type)*.

### Salida

**Recomendaciones:** Ranking de estaciones ordenadas por beneficio económico e impacto ambiental esperado, compatibles con el vehículo y con hábitos eficientes.

**Consejos de hábitos específicos** *(p. ej. “reduce ralentí >60s”, “suaviza aceleraciones”)*.


## 🧮 Ranking de repostajes (simple + ML opcional)

<!-- ###  Baseline (simple, explicable)

#### 📌 Puntuación por estación *s*

**Fórmula:**
scoreₛ = wₚ · (Δprecio · litros) − wₖₘextra · (km extra · €/km) − wₜ · (β · min espera) − wCO₂ · ΔCO₂


#### 📐 Definiciones

- **Δprecio** = precio_media_zona − priceₛ  
- **ΔCO₂** = litros × EFcombustible (g/l → kg)  
- **Pesos:**  
  - `wₚ`, `wₖₘextra`, `wₜ`, `wCO₂` son configurables según la política de sostenibilidad del cliente

#### ✅ Ventajas

- 100% trazable  
- Evaluación “what-if” instantánea (por ejemplo, subir `wCO₂` si el cliente prioriza reducir huella) -->


### 🧠 Modelo ML (learning-to-rank **LTR**, sencilla y escalable)

**Modelo:** LightGBM Ranker (LambdaRank) (https://lightgbm.readthedocs.io/en/latest/LambdaRank.html)  
**Objetivo:** Aprender la preferencia histórica de la flota o del propio cliente.

#### 🔍 Features
- Δprecio  
- km extra (Haversine)  
- espera  
- marca (one-hot)  
- distancia al destino  
- huella CO₂ por litro del vehículo  
- perfil de hábitos (ver sección 2.2)

#### 🧪 Agrupación
- Cada “consulta” (intento de repostaje) es un grupo de estaciones candidatas

#### ✅ Relevancia
- `1` si fue elegida históricamente  
- `0` si no (o ahorro neto descontado)

#### 📊 Métrica
- NDCG@k

#### 🌱 Sostenibilidad
Incluimos **ΔCO₂** (o un costo sombra de CO₂) como feature/etiqueta,  
de modo que el modelo **priorice estaciones con menor impacto** cuando  
el ahorro económico no compite con el costo ambiental.


## 🚗 Hábitos eficientes (eco-driving)

<!-- ### 🧭 Baseline (reglas simples)
- `% ralentí > umbral` → “apaga motor en paradas >60s”
- `% aceleración agresiva > umbral` → “suaviza aceleraciones”
- `% exceso de velocidad > umbral` → “usa control de crucero”
- `Varianza de velocidad alta` → “mantén velocidad estable” -->

### 🤖 Opción ML (simple y explicable)
- Clasificación “eficiente vs. ineficiente”  
  - **Target:** L/100km normalizado por contexto  
  - **Modelos:** RandomForest o Logistic Regression (coeficientes interpretables)

- Clustering (KMeans)  
  - Para segmentar estilos y priorizar consejos  
  - Ranking por impacto esperado

- Explicabilidad  
  - Usar **feature importances** o **coeficientes** para justificar cada consejo