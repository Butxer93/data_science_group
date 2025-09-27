# Plataforma de Recomendaciones: Repostaje + Hábitos Eficientes

API unificada (FastAPI) con dos módulos:
- **Repostaje**: Learning-to-Rank (LightGBM LambdaRank o fallback RF). Ranking de estaciones por coste y sostenibilidad (precio, desvío, espera, beneficios).
- **Hábitos (eco-driving)**: Clasificación (RandomForest con Grid opcional) + Clustering (KMeans con k por silhouette). Consejos por perfil.

## Requisitos
- Python 3.12
- Gestor de entorno: `uv`

## Arranque
```bash
uv sync
uv run fastapi dev app/api.py --reload-dir .
# docs:  http://localhost:8000/docs
# redoc: http://localhost:8000/redoc
# home:  http://localhost:8000/
```

##  Flujo sugerido:

- **Generar datos sintéticos:** `POST /repostaje/data/generar_sintetico`

- **Entrenar ranker:** `POST /repostaje/ml/entrenar_ranker?syntetico=true&optimized=true`

- **Entrenar hábitos:** `POST /habitos/ml/entrenar?optimized=true`

- **Pedir ranking:** `POST /repostaje/recomendaciones`

- **Predicción hábitos:** `POST /habitos/predict`