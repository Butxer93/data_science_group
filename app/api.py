# app/api.py
# Python 3.12 – API unificada: /repostaje/* y /habitos/*
from pathlib import Path
from typing import Optional, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ConfigDict, validator

# --- Núcleo negocio ---
from src.utils import (
    generar_todo_sintetico,
    generar_polilinea_desde_rutas,
    distancia_a_polilinea_km,
)
from src.rank_repostaje import rankear_candidatos
from src.train_ranker import construir_entrenamiento_sintetico, ajustar_pipeline
from src.model_store import (
    cargar_ranker_repostaje,
    guardar_ranker_repostaje,
    pesos_por_politica,
    cargar_modelos_habitos,
)
from src.habits_efficiency import entrenar_habitos, FEATURES_HABITOS, generar_telemetria_sintetica

# ------------------------------------------
# Config y metadatos OpenAPI
# ------------------------------------------
API_TITLE = "API Unificada: Repostaje + Hábitos"
API_VERSION = "1.0.1"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sinteticos"
MODELS_DIR = ROOT / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TAGS_METADATA = [
    {
        "name": "general",
        "description": "Endpoints de estado y metainformación de la API unificada.",
    },
    {
        "name": "repostaje",
        "description": (
            "Módulo de **ranking de estaciones**. Genera datos sintéticos, entrena un modelo LTR y devuelve recomendaciones."
        ),
    },
    {
        "name": "habitos",
        "description": (
            "Módulo de **hábitos eficientes** (clasificación + clustering) y consejos de eco-driving."
        ),
    },
]
OPENAPI_DESCRIPTION = """
Bienvenido a la **API Unificada** para *Repostaje* y *Hábitos eficientes*.

- **Repostaje**: genera datos, entrena y recomienda estaciones cerca de una ruta optimizando **coste** y **sostenibilidad** (precio, desvío, espera, tarjetas).
- **Hábitos**: entrena modelos de eco-driving y devuelve **consejos** personalizados.

> Requisitos: Python 3.12, `uv`.  
> Documentación: **/docs** (Swagger) y **/redoc** (ReDoc).
""".strip()

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    openapi_tags=TAGS_METADATA,
    description=OPENAPI_DESCRIPTION,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ------------------------------------------
# Utilidades locales
# ------------------------------------------
# def _leer_csv(nombre: str) -> pd.DataFrame:
#     p = DATA_DIR / nombre
#     if not p.exists():
#         return pd.DataFrame()
#     try:
#         return pd.read_csv(p)
#     except Exception as e:
#         raise HTTPException(400, f"Error leyendo {nombre}: {e}") from e

def _normalizar_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna(how="all").copy()
    df.columns = df.columns.map(lambda s: str(s).strip())
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
    return df

def _leer_csv(nombre: str) -> pd.DataFrame:
    p = DATA_DIR / nombre
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, skip_blank_lines=True)
        return _normalizar_df(df)
    except Exception as e:
        raise HTTPException(400, f"Error leyendo {nombre}: {e}") from e

def _enriquecer_sintetico_minimo(puntos: pd.DataFrame) -> pd.DataFrame:
    df = puntos.copy()
    if "precio_litro" not in df.columns:
        df["precio_litro"] = (1.55 + (df.index % 13) * 0.005).clip(1.40, 1.95)
    if "minutos_espera" not in df.columns:
        df["minutos_espera"] = (df.index % 9) + 1
    if "tipo_combustible" not in df.columns:
        df["tipo_combustible"] = ["diesel" if i % 2 == 0 else "gasolina" for i in range(len(df))]
    return df

# ------------------------------------------
# Modelos de entrada
# ------------------------------------------
Prioridad = Literal["coste", "equilibrado", "sostenible"]
TipoCliente = Literal["particular", "flota"]

class ReqRepostaje(BaseModel):
    ruta_id: str = Field(..., description="Identificador de la ruta en rutas.csv")
    litros_necesarios: float = Field(..., gt=0, description="Litros a repostar")
    precio_area_medio: float = Field(..., gt=0, description="Precio medio de la zona (€/L)")
    prioridad: Prioridad = Field("equilibrado")
    tipo_cliente: TipoCliente = Field("particular")
    tarjeta: Optional[str] = Field(None, description="Nombre de tarjeta en beneficios_tarjeta.csv")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "ruta_id": "R001",
                    "litros_necesarios": 45,
                    "precio_area_medio": 1.62,
                    "prioridad": "equilibrado",
                    "tipo_cliente": "flota",
                    "tarjeta": "MiTarjeta"
                }
            ]
        }
    )

# ==========================================
# Home / Health
# ==========================================
@app.get("/", response_class=HTMLResponse, tags=["general"])
def home():
    return f"""
    <html><head><meta charset="utf-8"/><title>{API_TITLE}</title></head>
    <body style="font-family:system-ui;margin:2rem;">
      <h1>🚀 {API_TITLE}</h1>
      <p>{OPENAPI_DESCRIPTION}</p>
      <p><a href="/docs">Swagger</a> · <a href="/redoc">ReDoc</a> · <a href="/health">/health</a></p>
    </body></html>
    """

@app.get("/health", tags=["general"])
def health():
    # repostaje
    pipe_rank = cargar_ranker_repostaje(MODELS_DIR)
    puntos = _leer_csv("puntos_repostaje.csv")
    rutas = _leer_csv("rutas.csv")
    # hábitos
    cls_h, clu_h, _rules = cargar_modelos_habitos(MODELS_DIR)
    return {
        "ok": True,
        "version": API_VERSION,
        "repostaje": {
            "modelo_cargado": bool(pipe_rank),
            "puntos_repostaje": int(len(puntos)) if not puntos.empty else 0,
            "rutas": int(len(rutas)) if not rutas.empty else 0,
        },
        "habitos": {
            "clasificador": cls_h is not None,
            "clustering": clu_h is not None,
            "features": FEATURES_HABITOS,
        },
    }

@app.get("/repostaje/data/rutas", tags=["repostaje"])
def listar_rutas(limit: int = Query(50, ge=1, le=1000)):
    df_rutas = _leer_csv("rutas.csv")
    if df_rutas.empty:
        raise HTTPException(404, "No hay rutas en data/sinteticos/rutas.csv")
    ids = sorted(df_rutas["ruta_id"].astype(str).str.strip().unique().tolist())
    ejemplo = df_rutas.head(min(limit, len(df_rutas))).to_dict(orient="records")
    return {"n_rutas": len(ids), "ruta_ids": ids[:50], "preview": ejemplo}

# ==========================================
# Repostaje: datos & entrenamiento
# ==========================================
@app.post("/repostaje/data/generar_sintetico", tags=["repostaje"])
def generar_sintetico_endpoint(
    n_puntos: int = Query(350, ge=50, le=100000, description="Filas para puntos_repostaje.csv"),
    n_rutas: int = Query(5, ge=1, le=1000, description="Número de rutas"),
    puntos_por_ruta: int = Query(70, ge=5, le=5000, description="Puntos por ruta"),
    n_ev: int = Query(220, ge=50, le=200000, description="Filas para puntos_recarga_electrica.csv"),
    n_tarjetas: int = Query(6, ge=1, le=5000, description="Tarjetas de beneficios"),
):
    summary = generar_todo_sintetico(DATA_DIR, n_puntos, n_rutas, puntos_por_ruta, n_ev, n_tarjetas)
    return {"ok": True, "archivos": summary, "ruta": str(DATA_DIR)}

@app.post("/repostaje/ml/entrenar_ranker", tags=["repostaje"])
def entrenar_ranker(
    syntetico: bool = Query(True, description="Usar generador interno LTR sintético"),
    optimized: bool = Query(False, description="Grid + GroupKFold con métrica NDCG@5"),
    n_consultas: int = Query(250, ge=50, le=5000),
    candidatos_por_consulta: int = Query(6, ge=2, le=50),
):
    puntos = _leer_csv("puntos_repostaje.csv")
    if puntos.empty:
        raise HTTPException(400, "data/sinteticos/puntos_repostaje.csv no encontrado. Usa /repostaje/data/generar_sintetico primero.")
    puntos = _enriquecer_sintetico_minimo(puntos)

    if syntetico:
        df_train = construir_entrenamiento_sintetico(
            puntos, n_consultas=n_consultas, cand_por_q=candidatos_por_consulta
        )
    else:
        df_train = construir_entrenamiento_sintetico(
            puntos, n_consultas=n_consultas, cand_por_q=candidatos_por_consulta
        )

    pipe, meta = ajustar_pipeline(df_train, optimized=optimized)
    guardar_ranker_repostaje(pipe, MODELS_DIR)
    return {
        "ok": True,
        "filas_entrenamiento": int(len(df_train)),
        "meta": meta,
        "salida_modelo": str(MODELS_DIR / "rank_repostaje_pipe.joblib"),
    }

@app.get("/repostaje/ml/info", tags=["repostaje"])
def ml_info_repostaje():
    pipe = cargar_ranker_repostaje(MODELS_DIR)
    if not pipe:
        return {"ok": True, "modelo_cargado": False}
    try:
        steps = [name for name, _ in pipe.steps]
    except Exception:
        steps = []
    return {"ok": True, "modelo_cargado": True, "pipeline_steps": steps}

# ==========================================
# Repostaje: recomendaciones
# ==========================================
class ReqRecomendacion(ReqRepostaje):
    pass

@app.post("/repostaje/recomendaciones", tags=["repostaje"])
def recomendaciones_repostaje(payload: ReqRecomendacion):
    df_rutas = _leer_csv("rutas.csv")
    if df_rutas.empty:
        raise HTTPException(400, "data/sinteticos/rutas.csv no encontrado. Genera datos primero.")

    # normaliza
    df_rutas["ruta_id"] = df_rutas["ruta_id"].astype(str).str.strip()
    payload_ruta = str(payload.ruta_id).strip()

    ids = set(df_rutas["ruta_id"].unique().tolist())
    if payload_ruta not in ids:
        # ayuda al usuario mostrando ejemplos
        ejemplos = sorted(list(ids))[:10]
        raise HTTPException(
            400,
            f"ruta_id={payload_ruta} no existe en rutas.csv. "
            f"Ejemplos válidos: {ejemplos}"
        )

    poli = generar_polilinea_desde_rutas(payload.ruta_id, df_rutas)
    if not poli:
        raise HTTPException(400, f"Ruta {payload.ruta_id} sin puntos válidos")

    puntos = _leer_csv("puntos_repostaje.csv")
    if puntos.empty:
        raise HTTPException(400, "data/sinteticos/puntos_repostaje.csv no encontrado.")
    puntos = _enriquecer_sintetico_minimo(puntos)
    beneficios = _leer_csv("beneficios_tarjeta.csv")

    # filtro ≤10km
    cand = []
    for _, s in puntos.iterrows():
        detour = distancia_a_polilinea_km((float(s["latitud"]), float(s["longitud"])), poli)
        if detour <= 10.0:
            d = s.to_dict()
            d["desvio_km"] = detour
            cand.append(d)
    if not cand:
        return {"estaciones": [], "prioridad": payload.prioridad, "modelo_usado": False, "motivo": "Sin candidatos <=10km"}

    ctx = {
        "litros_necesarios": float(payload.litros_necesarios),
        "precio_area_medio": float(payload.precio_area_medio),
        "eur_por_km": 0.18,
        "beta_espera": 0.05,
        "tipo_cliente": payload.tipo_cliente,
        "tarjeta": payload.tarjeta,
    }
    ctx.update(pesos_por_politica(payload.prioridad))

    pipe = cargar_ranker_repostaje(MODELS_DIR)  # puede ser None
    ranked = rankear_candidatos(
        candidatos=cand,
        contexto=ctx,
        df_beneficios=beneficios if not beneficios.empty else None,
        pipe=pipe,
    )

    cols_resp = [
        "punto_id", "nombre_estacion", "marca", "carretera", "latitud", "longitud",
        "tipo_combustible", "precio_litro", "precio_neto", "minutos_espera",
        "desvio_km", "score"
    ]
    estaciones = [
        {k: v for k, v in r.items() if k in cols_resp} | {"punto_id": r.get("punto_id", None)}
        for r in ranked
    ]
    return {
        "prioridad": payload.prioridad,
        "modelo_usado": bool(pipe),
        "n_candidatos": len(estaciones),
        "estaciones": estaciones[:50],
    }

# ==========================================
# Hábitos: entrenamiento y predicción
# ==========================================

@app.post("/habitos/data/generar_sintetico", tags=["habitos"])
def habitos_generar_sintetico(
    N: int = Query(5000, ge=500, le=200000),
    seed: int = Query(42, ge=0, le=1_000_000)
):
    """
    Genera y guarda `data/sinteticos/habitos_telemetria.csv`.
    """
    csv_path = generar_telemetria_sintetica(DATA_DIR, N=N, seed=seed)
    return {"ok": True, "csv": str(csv_path), "rows": int(pd.read_csv(csv_path).shape[0])}

@app.get("/habitos/data/telemetria", tags=["habitos"])
def habitos_preview(limit: int = Query(20, ge=1, le=200)):
    """
    Devuelve un preview del dataset guardado (si existe).
    """
    p = DATA_DIR / "habitos_telemetria.csv"
    if not p.exists():
        return {"ok": False, "error": "No existe data/sinteticos/habitos_telemetria.csv. Genera primero con /habitos/data/generar_sintetico."}
    df = pd.read_csv(p)
    return {"ok": True, "rows": int(len(df)), "features": FEATURES_HABITOS, "preview": df.head(limit).to_dict(orient="records")}

@app.post("/habitos/ml/entrenar", tags=["habitos"])
def entrenar_habitos_endpoint(
    optimized: bool = Query(False, description="Grid RF + selección de k por silhouette"),
    reuse_if_exists: bool = Query(True, description="Si el CSV de hábitos existe, reutilizarlo"),
    N: int = Query(5000, ge=500, le=200000),
    seed: int = Query(42, ge=0, le=1_000_000)
):
    """
    Entrena hábitos reutilizando (si existe) o generando el dataset `habitos_telemetria.csv`.
    """
    clf_bundle, clu_bundle, reglas_path, meta = entrenar_habitos(
        models_dir=MODELS_DIR,
        optimized=optimized,
        data_dir=DATA_DIR,
        reuse_if_exists=reuse_if_exists,
        N=N,
        seed=seed,
        persist_rules=True
    )
    return {"ok": True, "features": FEATURES_HABITOS, "reglas": str(reglas_path), "meta": meta}

@app.post("/habitos/predict", tags=["habitos"])
def predecir_habitos(payload: dict):
    feats = [payload.get(k) for k in FEATURES_HABITOS]
    if any(v is None for v in feats):
        raise HTTPException(400, f"Faltan variables: {FEATURES_HABITOS}")

    cls, clu, reglas = cargar_modelos_habitos(MODELS_DIR)
    if cls is None:
        raise HTTPException(503, "Clasificador no entrenado. Llama a /habitos/ml/entrenar")

    eff = int(cls["model"].predict([feats])[0])
    prob = None
    try:
        prob = float(cls["model"].predict_proba([feats])[0][1])
    except Exception:
        pass

    cluster_id, tips = None, []
    if clu is not None:
        cluster_id = int(clu["pipeline"].predict([feats])[0])
        tips = (reglas.get("rules_by_cluster", {}).get(str(cluster_id), [])
                or reglas.get("rules_by_cluster", {}).get(cluster_id, []))

    return {"eficiente": eff, "prob_eficiente": prob, "cluster": cluster_id, "consejos": tips}
