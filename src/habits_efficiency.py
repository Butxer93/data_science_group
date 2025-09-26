import numpy as np, pandas as pd, json
from pathlib import Path
from typing import Tuple, Dict, Any, List

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

import joblib

from .model_store import guardar_modelos_habitos, cargar_modelos_habitos

# ----- Config rutas robustas (sin depender de constantes importadas) -----
ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
HABITS_RULES_PATH = MODELS_DIR / "habits_rules.json"

FEATURES_HABITOS = [
    "velocidad_media_kmh",
    "frenadas_fuertes_100km",
    "aceleraciones_100km",
    "ratio_ralenti",
    "ratio_carga",
]

# ---------------------- Datos sintéticos ----------------------
def _sintetico_telemetria(N=5000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "vehiculo_id": rng.choice(["V001","V002","V003","V004"], size=N, p=[0.4,0.3,0.2,0.1]),
        "velocidad_media_kmh": np.clip(rng.normal(68, 12, N), 20, 120),
        "frenadas_fuertes_100km": np.clip(rng.normal(3.5, 2.0, N), 0, 15),
        "aceleraciones_100km": np.clip(rng.normal(5.0, 2.5, N), 0, 20),
        "ratio_ralenti": np.clip(rng.normal(0.08, 0.05, N), 0, 0.6),
        "ratio_carga": np.clip(rng.normal(0.45, 0.2, N), 0, 1.0)
    })
    # consumo sintético y etiqueta eficiente
    base = 6.5
    df["consumo_l_100km"] = (base
        + 0.03*(df["velocidad_media_kmh"]-70)**2/100
        + 0.4*df["ratio_ralenti"]*10
        + 0.08*df["frenadas_fuertes_100km"]
        + 0.05*df["aceleraciones_100km"]
        + 1.5*df["ratio_carga"])
    df["eficiente"] = 0
    for vid, grp in df.groupby("vehiculo_id"):
        thr = np.percentile(grp["consumo_l_100km"], 40)
        df.loc[df["vehiculo_id"]==vid, "eficiente"] = (df.loc[df["vehiculo_id"]==vid, "consumo_l_100km"] <= thr).astype(int)
    return df

# ---------------------- Clasificador ----------------------
def _cv_score_rf(
    X: np.ndarray, y: np.ndarray, params: Dict[str, Any], n_splits: int = 5, seed: int = 42
) -> float:
    """F1 macro medio en CV estratificado."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: List[float] = []
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(
            random_state=42, class_weight="balanced", **params
        )
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        scores.append(f1_score(y[te], pred, average="macro"))
    return float(np.mean(scores))

def _grid_search_rf(X: np.ndarray, y: np.ndarray) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    grid = [
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 400, "max_depth": 20,   "min_samples_leaf": 2},
        {"n_estimators": 400, "max_depth": 30,   "min_samples_leaf": 2},
        {"n_estimators": 300, "max_depth": 25,   "min_samples_leaf": 4},
    ]
    best_score, best_params = -1.0, None
    for params in grid:
        score = _cv_score_rf(X, y, params)
        if score > best_score:
            best_score, best_params = score, params
    # Entrenamiento final con mejores params
    best_clf = RandomForestClassifier(
        random_state=42, class_weight="balanced", **best_params
    )
    best_clf.fit(X, y)
    return best_clf, {"f1_macro_cv": round(best_score, 4), "params": best_params}

# ---------------------- Clustering ----------------------
def _select_kmeans(X: np.ndarray, k_values=(3,4,5,6)) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Selecciona k por silhouette score con StandardScaler + KMeans.
    """
    best_sil, best_k, best_pipe = -1.0, None, None
    for k in k_values:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=42))
        ])
        labels = pipe.fit_predict(X)
        # Evitar casos degenerados (todos en el mismo cluster)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(X, labels)
        if sil > best_sil:
            best_sil, best_k, best_pipe = sil, k, pipe
    # Si por alguna razón no se pudo, fallback a k=4
    if best_pipe is None:
        best_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=4, n_init=10, random_state=42))
        ]).fit(X)
        best_k, best_sil = 4, float("nan")
    return best_pipe, {"k": best_k, "silhouette": None if np.isnan(best_sil) else round(best_sil, 4)}

def _build_rules_from_clusters(df: pd.DataFrame, labels: np.ndarray) -> Dict[int, List[str]]:
    dfc = df.copy()
    dfc["_cluster"] = labels
    perfiles = dfc.groupby("_cluster")[FEATURES_HABITOS + ["consumo_l_100km","eficiente"]].mean()
    rules_map = {}
    for c in sorted(perfiles.index):
        p = perfiles.loc[c]
        tips = []
        if p["ratio_ralenti"] > 0.12: tips.append("Reducir ralentí (>12%) con apagado en esperas >2 min")
        if p["frenadas_fuertes_100km"] > 5: tips.append("Anticipar frenadas (evitar >5/100km)")
        if p["aceleraciones_100km"] > 7: tips.append("Acelerar progresivo (objetivo <7/100km)")
        if p["velocidad_media_kmh"] > 90: tips.append("Mantener crucero entre 70–90 km/h")
        if p["ratio_carga"] > 0.7: tips.append("Optimizar carga (consolidar envíos)")
        if not tips:
            tips = ["Buenos hábitos; mantener formación"]
        rules_map[int(c)] = tips
    return rules_map

# ---------------------- API-facing: entrenar ----------------------
def entrenar_habitos(optimized: bool = False):
    """
    Entrena clasificador y clustering con datos sintéticos.
    - optimized=False: RF por defecto (n_estimators=300), KMeans k=4
    - optimized=True: grid RF (mejor F1 macro CV) y selección k (mejor silhouette)
    Devuelve (clf_bundle, clu_bundle, reglas_path, meta)
    """
    df = _sintetico_telemetria()
    X = df[FEATURES_HABITOS].values
    y = df["eficiente"].values

    meta = {}

    # Clasificador
    if optimized:
        clf, info = _grid_search_rf(X, y)
        meta["clf"] = {"strategy": "grid_cv", **info}
    else:
        clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
        clf.fit(X, y)
        meta["clf"] = {"strategy": "default", "params": {"n_estimators": 300}}

    clf_bundle = {"model": clf, "features": FEATURES_HABITOS, "target_name": "eficiente", "version":"0.2.0"}

    # Clustering
    if optimized:
        clu_pipe, info = _select_kmeans(X, k_values=(3,4,5,6))
        meta["cluster"] = {"strategy":"silhouette", **info}
    else:
        clu_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=4, n_init=10, random_state=42))
        ]).fit(X)
        meta["cluster"] = {"strategy":"default", "k":4}

    clu_bundle = {"pipeline": clu_pipe, "features": FEATURES_HABITOS, "version":"0.2.0"}

    # Reglas de negocio
    labels = clu_pipe.predict(X)
    rules_map = _build_rules_from_clusters(df, labels)
    HABITS_RULES_PATH.write_text(
        json.dumps({"rules_by_cluster": rules_map}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Guardado (para que la API las pueda cargar)
    guardar_modelos_habitos(clf_bundle, clu_bundle)

    return clf_bundle, clu_bundle, HABITS_RULES_PATH, meta

# Compat: endpoint “rápido” previo
def entrenar_habitos_sintetico():
    return entrenar_habitos(optimized=False)