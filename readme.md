# Desafio_R3 — uv + FastAPI (2 servicios) + ML (LTR & Hábitos)

## Requisitos
- Python 3.12
- uv: https://docs.astral.sh/uv/

## Instalar dependencias
```bash
uv sync

```
## Generar datos sintétios

```bash
uv run python scripts/generar_datos.py
```

Esto creará:

- data/puntos_repostaje.csv (≈350 filas)

- data/rutas.csv (5 rutas × 70 puntos = 350 filas)

- data/puntos_recarga_electrica.csv (≈220 filas)

- data/beneficios_tarjeta.csv (≈6 filas)

## Ejecutar servicio de Repostaje (ranking combustible)

```bash

uv run fastapi dev app_repostaje/api.py
# o producción simple:
uv run fastapi run app_repostaje.api:app --host 0.0.0.0 --port 8000
# docs: http://localhost:8000/docs
```

## Ejecutar servicio de Hábitos (ranking eficiencia)

```bash
uv run fastapi dev app_habitos/api.py
# docs: http://localhost:8001/docs (usa --port 8001 en run/dev )
```

