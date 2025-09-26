import numpy as np, pandas as pd, json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from src.ml.model_store import guardar_modelos_habitos, cargar_modelos_habitos, HABITS_RULES as HABITS_RULES_PATH, MODELS as MODELS_DIR

FEATURES_HABITOS = [
    "velocidad_media_kmh",
    "frenadas_fuertes_100km",
    "aceleraciones_100km",
    "ratio_ralenti",
    "ratio_carga"
]

def _sintetico_telemetria(N=5000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "vehiculo_id": rng.choice(["V001","V002","V003","V004"], size=N, p=[0.4,0.3,0.2,0.1]),
        "velocidad_media_kmh": np.clip(rng.normal(68, 12, N), 20, 120),
        "frenadas_fuertes_100km": np.clip(rng.normal(3.5, 2.0, N), 0, 15),
        "aceleraciones_100km": np.clip(rng.normal(5.0, 2.5, N), 0, 20),
        "ratio_ralenti": np.clip(rng.normal(0.08, 0.05, N), 0, 0.6),
        "ratio_carga": np.clip(rng.norimal(0.45, 0.2, N), 0, 1.0)
    })
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

def entrenar_habitos_sintetico():
    df = _sintetico_telemetria()
    X = df[FEATURES_HABITOS].values
    y = df["eficiente"].values

    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    clf.fit(X, y)
    clf_bundle = {"model": clf, "features": FEATURES_HABITOS, "target_name":"eficiente", "version":"0.1.0"}

    cluster_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=4, n_init=10, random_state=42))
    ])
    cluster_pipe.fit(X)
    clu_bundle = {"pipeline": cluster_pipe, "features": FEATURES_HABITOS, "version":"0.1.0"}

    # Reglas por cluster (promedios)
    labels = cluster_pipe.predict(X)
    df["_cluster"] = labels
    perfiles = df.groupby("_cluster")[FEATURES_HABITOS + ["consumo_l_100km","eficiente"]].mean()

    rules_map = {}
    for c in sorted(perfiles.index):
        p = perfiles.loc[c]
        tips = []
        if p["ratio_ralenti"] > 0.12: tips.append("Reducir ralentí (>12%) con apagado en esperas >2 min")
        if p["frenadas_fuertes_100km"] > 5: tips.append("Anticipar frenadas (evitar >5/100km)")
        if p["aceleraciones_100km"] > 7: tips.append("Acelerar progresivo (objetivo <7/100km)")
        if p["velocidad_media_kmh"] > 90: tips.append("Crucero entre 70–90 km/h")
        if p["ratio_carga"] > 0.7: tips.append("Optimizar carga (consolidar envíos)")
        if not tips:
            tips = ["Buenos hábitos; mantener formación"]
        rules_map[int(c)] = tips

    MODELS.mkdir(parents=True, exist_ok=True)
    HABITS_RULES.write_text(json.dumps({"rules_by_cluster": rules_map}, ensure_ascii=False, indent=2), encoding="utf-8")

    return clf_bundle, clu_bundle, HABITS_RULES
