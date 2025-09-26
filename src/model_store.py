from pathlib import Path
import os, joblib, json

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

# Rutas de artefactos
RANKER_REPOSTAJE = MODELS / "rank_repostaje_pipe.joblib"
HABITS_CLS = MODELS / "habits_classifier.joblib"
HABITS_CLU = MODELS / "habits_clusters.joblib"
HABITS_RULES = MODELS / "habits_rules.json"

def cargar_ranker_repostaje():
    return joblib.load(RANKER_REPOSTAJE) if RANKER_REPOSTAJE.exists() else None

def guardar_ranker_repostaje(pipe):
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, RANKER_REPOSTAJE)

def pesos_por_politica(prioridad: str):
    # pesos y precio sombra de CO2: coste vs sostenible
    if prioridad == "sostenible":
        return {"w_ahorro":1.0, "w_desvio":1.0, "w_espera":1.0, "w_co2":1.0, "precio_sombra_co2": 0.2}
    if prioridad == "coste":
        return {"w_ahorro":1.0, "w_desvio":1.0, "w_espera":1.0, "w_co2":0.2, "precio_sombra_co2": 0.0}
    return {"w_ahorro":1.0, "w_desvio":1.0, "w_espera":1.0, "w_co2":0.5, "precio_sombra_co2": 0.1}

# Hábitos
def cargar_modelos_habitos():
    cls = joblib.load(HABITS_CLS) if HABITS_CLS.exists() else None
    clu = joblib.load(HABITS_CLU) if HABITS_CLU.exists() else None
    reglas = json.loads(HABITS_RULES.read_text(encoding="utf-8")) if HABITS_RULES.exists() else {"rules_by_cluster": {}}
    return cls, clu, reglas

def guardar_modelos_habitos(clf_bundle, clu_bundle):
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf_bundle, HABITS_CLS)
    joblib.dump(clu_bundle, HABITS_CLU)
