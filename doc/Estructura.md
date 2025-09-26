## ✅ Desafio_R3

- app/api.py → muy claro: aquí va la FastAPI.

- data/ → CSVs de entrada.

- models/ → almacenamiento de artefactos .joblib y .json.

- scr/ml/ → separación para código de Machine Learning (ranking, entrenamiento, almacenamiento).

- scr/utils.py → utilidades genéricas.


Desafio_R3/
├─ pyproject.toml
├─ README.md
├─ app_repostaje/
│  ├─ __init__.py
│  └─ api.py
├─ app_habitos/
│  ├─ __init__.py
│  └─ api.py
├─ src/
│  ├─ __init__.py
│  ├─ utils.py
│  └─ ml/
│     ├─ __init__.py
│     ├─ rank_repostaje.py
│     ├─ train_ranker.py
│     ├─ habits_efficiency.py
│     └─ model_store.py
├─ data/
│  ├─ puntos_repostaje.csv
│  ├─ rutas.csv
│  ├─ puntos_recarga_electrica.csv
│  └─ beneficios_tarjeta.csv
└─ models/              # artefactos .joblib / .json