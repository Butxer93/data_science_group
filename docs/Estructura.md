## ✅ Desafio_R3

- app/api.py → muy claro: aquí va la FastAPI.

- data/ → CSVs de entrada.

- models/ → almacenamiento de artefactos .joblib y .json.

- scr/ml/ → separación para código de Machine Learning (ranking, entrenamiento, almacenamiento).

- scr/utils.py → utilidades genéricas.


## Notas

- La “grid” es pequeña y robusta (no dependemos de lightgbm.sklearn.GridSearchCV con group interno, que suele dar guerra).
Hacemos loop manual por combinación y validación GroupKFold con métrica NDCG@5 por consulta.

- Para fallback RandomForest, el score también es NDCG@5 usando la etiqueta binaria rel para evaluar el orden de y_pred.

