# syntax=docker/dockerfile:1.7

# Imagen base mínima con Python 3.12
FROM python:3.12-slim AS base

# Evitar prompts y mejorar pip
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/root/.cache/uv

# Paquetes del sistema necesarios (libgomp1 para lightgbm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv (gestor de entornos) – bin en /root/.cargo/bin/uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copia solo los archivos de definición para aprovechar la caché de dependencias
COPY pyproject.toml README.md ./

# Sin lockfile: resolvemos y creamos el entorno (solo deps runtime)
RUN uv sync --no-dev --python /usr/local/bin/python

# Copiamos el resto del proyecto
COPY app ./app
COPY src ./src
# Estructura de datos y modelos (se montarán como volúmenes, pero creamos los paths)
RUN mkdir -p data/sinteticos data/raw models notebooks informes

# Puerto de la API
EXPOSE 8000

# Usuario no root por seguridad
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Comando por defecto: Uvicorn con 2 workers
# (fastapi run también funciona, pero aquí ajustamos workers explícitos)
CMD ["uv", "run", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
