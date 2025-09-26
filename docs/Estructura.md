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



## Habitos
Clasificador (RandomForest) con búsqueda de hiperparámetros y validación StratifiedKFold (métrica: F1 macro).

Clustering (KMeans) con selección de k óptimo (entre 3–6) por silhouette score.

Endpoints FastAPI actualizados para activar la optimización con ?optimized=true.


## Notas y buenas prácticas

Métrica de selección del clasificador: F1 macro (balancea clases). Si prefieres ROC AUC, dime y te doy la variante.

Selección de k con silhouette sobre features crudas; al ir dentro de Pipeline(scaler+kmeans) se estandarizan automáticamente.

Se guardan artefactos en models/ (mismos nombres que usábamos), así el endpoint de predicción no cambia.

Si quieres reproducibilidad estricta, mantén random_state/seed tal y como están.