from __future__ import annotations
from pathlib import Path
import json
import joblib
from typing import Optional, Tuple

# -------------------------
# Rutas y guardado/carga
# -------------------------
def _ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def guardar_ranker_repostaje(pipe, models_dir: Path):
    models_dir = Path(models_dir)
    _ensure_dir(models_dir)
    joblib.dump(pipe, models_dir / "rank_repostaje_pipe.joblib")

def cargar_ranker_repostaje(models_dir: Path):
    models_dir = Path(models_dir)
    p = models_dir / "rank_repostaje_pipe.joblib"
    if not p.exists():
        return None
    return joblib.load(p)

def guardar_modelos_habitos(clf_bundle: dict, clu_bundle: dict, rules_path: Path, models_dir: Path):
    models_dir = Path(models_dir)
    _ensure_dir(models_dir)
    joblib.dump(clf_bundle, models_dir / "habits_classifier.joblib")
    joblib.dump(clu_bundle, models_dir / "habits_clusters.joblib")
    # rules ya se guardó en rules_path

def cargar_modelos_habitos(models_dir: Path) -> Tuple[Optional[dict], Optional[dict], dict]:
    models_dir = Path(models_dir)
    p1 = models_dir / "habits_classifier.joblib"
    p2 = models_dir / "habits_clusters.joblib"
    p3 = models_dir / "habits_rules.json"
    cls = joblib.load(p1) if p1.exists() else None
    clu = joblib.load(p2) if p2.exists() else None
    rules = json.loads(p3.read_text(encoding="utf-8")) if p3.exists() else {}
    return cls, clu, rules

# -------------------------
# Políticas de pesos
# -------------------------
def pesos_por_politica(prioridad: str) -> dict:
    # pesos para heurística (sólo si no hay modelo)
    if prioridad == "coste":
        return {"w_coste": 0.7, "w_desvio": 0.2, "w_tiempo": 0.1}
    if prioridad == "sostenible":
        return {"w_coste": 0.5, "w_desvio": 0.3, "w_tiempo": 0.2}
    return {"w_coste": 0.6, "w_desvio": 0.3, "w_tiempo": 0.1}
