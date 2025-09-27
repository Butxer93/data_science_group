# src/rank_repostaje.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

def _precio_neto_con_tarjeta(row: dict, tarjeta: Optional[str], df_beneficios: Optional[pd.DataFrame]) -> float:
    precio = float(row.get("precio_litro", 0.0))
    if not tarjeta or df_beneficios is None or df_beneficios.empty:
        return precio
    sub = df_beneficios[df_beneficios["tarjeta"].astype(str).str.lower() == str(tarjeta).lower()]
    if sub.empty:
        return precio
    cents = sub["centimos_litro"].astype(str).replace("", "0").astype(float).fillna(0.0).max()
    pct = sub["porcentaje"].astype(str).replace("", "0").astype(float).fillna(0.0).max() / 100.0
    # si hay estaciones explícitas, chequeo naive por nombre_estacion
    nombres = "|".join(sub["estaciones_incluidas"].fillna("").tolist())
    aplica = True
    if nombres.strip():
        aplica = any(n.strip() and n.strip() in str(row.get("nombre_estacion", "")) for n in nombres.split("|"))
    if not aplica:
        return precio
    precio = max(0.0, precio - cents/100.0)
    precio = precio * (1.0 - pct)
    return float(precio)

def _features_para_modelo(cand: List[dict], ctx: dict, df_beneficios: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for r in cand:
        pn = _precio_neto_con_tarjeta(r, ctx.get("tarjeta"), df_beneficios)
        rows.append({
            "marca": r.get("marca"),
            "tipo_combustible": r.get("tipo_combustible"),
            "carretera": r.get("carretera"),
            "delta_precio": float(ctx["precio_area_medio"]) - float(r.get("precio_litro", pn)),
            "desvio_km": float(r.get("desvio_km", 0.0)),
            "minutos_espera": float(r.get("minutos_espera", 0.0)),
            "litros_necesarios": float(ctx["litros_necesarios"]),
            "precio_neto": pn,
            "precio_litro": float(r.get("precio_litro", pn)),
            "nombre_estacion": r.get("nombre_estacion"),
            "punto_id": r.get("punto_id"),
            "latitud": r.get("latitud"),
            "longitud": r.get("longitud"),
        })
    return pd.DataFrame(rows)

def _heuristica_score(df: pd.DataFrame, ctx: dict, w_c=0.6, w_d=0.3, w_t=0.1) -> np.ndarray:
    ahorro = (ctx["precio_area_medio"] - df["precio_litro"].astype(float)) * ctx["litros_necesarios"]
    coste_desvio = df["desvio_km"].astype(float) * ctx.get("eur_por_km", 0.18)
    penal_tiempo = df["minutos_espera"].astype(float) * ctx.get("beta_espera", 0.05)
    score = w_c * ahorro - w_d * coste_desvio - w_t * penal_tiempo
    return score.values

def rankear_candidatos(candidatos: List[dict], contexto: dict,
                       df_beneficios: Optional[pd.DataFrame] = None,
                       pipe=None) -> List[Dict[str, Any]]:
    df = _features_para_modelo(candidatos, contexto, df_beneficios)
    if pipe is not None:
        try:
            X = df[["delta_precio","desvio_km","minutos_espera","litros_necesarios","marca","tipo_combustible","carretera"]]
            scores = pipe.predict(X)
        except Exception:
            scores = _heuristica_score(df, contexto)
    else:
        scores = _heuristica_score(df, contexto)
    df["score"] = scores
    # recomputar precio_neto en salida
    df["precio_neto"] = df.apply(lambda r: _precio_neto_con_tarjeta(r.to_dict(), contexto.get("tarjeta"), df_beneficios), axis=1)
    # devolver integrando algunos campos
    out = []
    for _, r in df.sort_values("score", ascending=False).iterrows():
        out.append({
            **{k: r.get(k) for k in ["punto_id","nombre_estacion","marca","carretera","latitud","longitud","tipo_combustible"]},
            "precio_litro": float(r["precio_litro"]),
            "precio_neto": float(r["precio_neto"]),
            "minutos_espera": float(r["minutos_espera"]),
            "desvio_km": float(r["desvio_km"]),
            "score": float(r["score"]),
        })
    return out
