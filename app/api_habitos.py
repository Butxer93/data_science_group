from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.habits_efficiency import (
    entrenar_habitos,
    entrenar_habitos_sintetico,  # compat
    cargar_modelos_habitos,      # vía model_store
    FEATURES_HABITOS
)

app = FastAPI(title="API Hábitos Eficientes (eco-driving)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@app.get("/health")
def health():
    cls, clu, _ = cargar_modelos_habitos()
    return {"ok": True, "clasificador": cls is not None, "clustering": clu is not None}

@app.post("/habitos/entrenar")
def entrenar_habitos_endpoint(optimized: bool = Query(False, description="Activa grid CV para RF y selección k por silhouette")):
    clf_bundle, clu_bundle, reglas_path, meta = entrenar_habitos(optimized=optimized)
    return {"ok": True, "features": FEATURES_HABITOS, "reglas": str(reglas_path), "meta": meta}

@app.post("/habitos/predict")
def predecir(payload: dict):
    feats = [payload.get(k) for k in FEATURES_HABITOS]
    if any(v is None for v in feats):
        raise HTTPException(400, f"Faltan variables: {FEATURES_HABITOS}")

    cls, clu, reglas = cargar_modelos_habitos()
    if cls is None:
        raise HTTPException(503, "Clasificador no entrenado. Llama a /habitos/entrenar")

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

    return {
        "eficiente": eff,
        "prob_eficiente": prob,
        "cluster": cluster_id,
        "consejos": tips
    }
