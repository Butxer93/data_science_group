from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd

from src.utils import distancia_a_polilinea_km, generar_polilinea_desde_rutas
from src.rank_repostaje import rankear_candidatos
from src.train_ranker import construir_entrenamiento_sintetico, ajustar_pipeline
from src.model_store import cargar_ranker_repostaje, guardar_ranker_repostaje, pesos_por_politica

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

app = FastAPI(title="API Repostaje (ranking combustible)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

def leer_csv(nombre: str) -> pd.DataFrame:
    p = DATA_DIR / nombre
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@app.get("/health")
def health():
    return {"ok": True, "modelo_cargado": cargar_ranker_repostaje() is not None}

# ...
@app.post("/train_ranker")
def entrenar_ranker(syntetico: bool = True, optimized: bool = False):
    puntos = leer_csv("puntos_repostaje.csv")
    if puntos.empty:
        raise HTTPException(400, "data/puntos_repostaje.csv no encontrado. Genera datos sintéticos primero.")
    df = construir_entrenamiento_sintetico(puntos, n_consultas=250, cand_por_q=6) if syntetico \
         else construir_entrenamiento_sintetico(puntos)
    pipe, meta = ajustar_pipeline(df, optimized=optimized)   # <<--- aquí
    guardar_ranker_repostaje(pipe)
    return {"ok": True, "meta": meta, "filas_entrenamiento": int(len(df))}


@app.post("/recomendaciones/repostaje")
def recomendaciones_repostaje(
    payload: dict
):
    """
    payload:
    {
      "ruta_id": "R001"  // o alternativamente start/end coords (no estricto aquí)
      "litros_necesarios": 45,
      "precio_area_medio": 1.62,
      "prioridad": "equilibrado",  // "coste" | "equilibrado" | "sostenible"
      "tipo_cliente": "particular", // "particular" | "flota"
      "tarjeta": "MiTarjeta"        // opcional, aplica a combustible
    }
    """
    ruta_id = payload.get("ruta_id")
    litros = float(payload.get("litros_necesarios", 40.0))
    precio_area = float(payload.get("precio_area_medio", 1.62))
    prioridad = payload.get("prioridad", "equilibrado")
    tipo_cliente = payload.get("tipo_cliente", "particular")
    tarjeta = payload.get("tarjeta")  # puede ser None

    df_rutas = leer_csv("rutas.csv")
    if df_rutas.empty:
        raise HTTPException(400, "data/rutas.csv no encontrado.")
    if ruta_id is None or ruta_id not in df_rutas["ruta_id"].unique():
        raise HTTPException(400, "ruta_id inválido o no presente en rutas.csv")

    # Polilínea de la ruta (ordenada por km_desde_origen)
    poli = generar_polilinea_desde_rutas(ruta_id, df_rutas)

    # Candidatos: puntos de combustible + beneficios de tarjeta
    puntos = leer_csv("puntos_repostaje.csv")
    if puntos.empty:
        raise HTTPException(400, "data/puntos_repostaje.csv no encontrado.")
    beneficios = leer_csv("beneficios_tarjeta.csv")

    # Enriquecimiento sintético si faltan campos (precio/minutos/tipo_combustible)
    if "precio_litro" not in puntos.columns:
        puntos["precio_litro"] = (1.50 + (puntos.index % 7) * 0.01).clip(1.35, 1.90)
    if "minutos_espera" not in puntos.columns:
        puntos["minutos_espera"] = (puntos.index % 10) + 1
    if "tipo_combustible" not in puntos.columns:
        puntos["tipo_combustible"] = ["diesel" if i % 2 == 0 else "gasolina" for i in range(len(puntos))]

    # Filtrado por desvío al corredor de ruta (10 km)
    candidatos = []
    for _, s in puntos.iterrows():
        detour = distancia_a_polilinea_km((s["latitud"], s["longitud"]), poli)
        if detour <= 10.0:
            candidatos.append({**s.to_dict(), "desvio_km": detour})

    if not candidatos:
        return {"estaciones": [], "prioridad": prioridad, "modelo_usado": False}

    # Contexto de scoring
    ctx = {
        "litros_necesarios": litros,
        "precio_area_medio": precio_area,
        "eur_por_km": 0.18,
        "beta_espera": 0.05,
        "tipo_cliente": tipo_cliente,
        "tarjeta": tarjeta
    }
    ctx.update(pesos_por_politica(prioridad))

    # Cargar modelo si existe
    pipe = cargar_ranker_repostaje()  # puede ser None (baseline)
    ranked = rankear_candidatos(
        candidatos=candidatos,
        contexto=ctx,
        df_beneficios=beneficios if not beneficios.empty else None,
        pipe=pipe
    )
    return {"estaciones": ranked, "prioridad": prioridad, "modelo_usado": bool(pipe)}

