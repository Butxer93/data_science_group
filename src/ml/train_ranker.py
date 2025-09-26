# src/ml/train_ranker.py
import pandas as pd, numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold

NUM = ["delta_precio","desvio_km","minutos_espera","litros_necesarios"]
CAT = ["marca","tipo_combustible","carretera"]

# ----------------- utilidades métrica -----------------
def ndcg_at_k(rel, k=5):
    rel = np.asfarray(rel)[:k]
    dcg = np.sum((2**rel - 1) / np.log2(np.arange(2, rel.size + 2)))
    rel_sorted = np.sort(rel)[::-1]
    idcg = np.sum((2**rel_sorted - 1) / np.log2(np.arange(2, rel_sorted.size + 2)))
    return dcg / idcg if idcg > 0 else 0.0

def eval_ndcg5_over_groups(pipe, X, rel_labels, groups):
    scores = pipe.predict(X)
    qids = groups
    ndcgs=[]
    for q in np.unique(qids):
        idx = np.where(qids == q)[0]
        order = np.argsort(-scores[idx])
        rel_q = rel_labels[idx][order]
        ndcgs.append(ndcg_at_k(rel_q, 5))
    return float(np.mean(ndcgs))

# ----------------- sintéticos -----------------
def _sintetizar_campos_faltantes(puntos: pd.DataFrame) -> pd.DataFrame:
    df = puntos.copy()
    if "precio_litro" not in df.columns:
        df["precio_litro"] = (1.50 + (df.index % 7) * 0.01).clip(1.35, 1.95)
    if "minutos_espera" not in df.columns:
        df["minutos_espera"] = (df.index % 10) + 1
    if "tipo_combustible" not in df.columns:
        df["tipo_combustible"] = ["diesel" if i % 2 == 0 else "gasolina" for i in range(len(df))]
    return df

def construir_entrenamiento_sintetico(puntos: pd.DataFrame, n_consultas=300, cand_por_q=6, seed=42):
    rng = np.random.default_rng(seed)
    puntos = _sintetizar_campos_faltantes(puntos)
    marcas = puntos["marca"].dropna().unique().tolist() or ["Repsol","Cepsa"]
    fuels  = puntos["tipo_combustible"].dropna().unique().tolist() or ["diesel","gasolina"]
    carrets= puntos["carretera"].dropna().unique().tolist() or ["A-1","A-2"]
    rows=[]
    for q in range(n_consultas):
        precio_area = float(np.clip(rng.normal(1.62, 0.04), 1.45, 1.95))
        litros = float(np.clip(rng.normal(45, 10), 12, 80))
        for _ in range(cand_por_q):
            marca = rng.choice(marcas); fuel = rng.choice(fuels); carr = rng.choice(carrets)
            precio = float(np.clip(precio_area + rng.normal(0,0.03), 1.40, 1.95))
            desvio = float(np.clip(abs(rng.normal(1.5, 1.0)), 0, 12))
            espera = float(np.clip(rng.normal(4,2.5), 0, 20))
            ahorro = max(0.0, precio_area - precio) * litros
            penal = desvio*0.18 + 0.05*espera + 0.1
            utilidad = ahorro - penal + (0.05 if marca in ["Repsol","Cepsa"] else 0.0)
            rows.append(dict(
                consulta_id=q,
                marca=marca, tipo_combustible=fuel, carretera=carr,
                precio_litro=precio, minutos_espera=espera, desvio_km=desvio,
                litros_necesarios=litros, precio_area_medio=precio_area,
                delta_precio=precio_area - precio, utilidad=utilidad
            ))
    df = pd.DataFrame(rows)
    df["rel"] = 0
    df.loc[df.groupby("consulta_id")["utilidad"].idxmax(), "rel"] = 1
    return df

# ----------------- grid optimizado -----------------
def _make_pre():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)
    ])

def _fit_lgbm_ranker_with_groups(X, y_rel, groups, params):
    import lightgbm as lgb
    model = lgb.LGBMRanker(
        objective="lambdarank",
        random_state=42,
        **params
    )
    pipe = Pipeline([("pre", _make_pre()), ("ranker", model)])
    pipe.fit(X, y_rel, ranker__group=groups)
    return pipe

def _fit_rf_regressor(X, y_util, params):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(random_state=42, **params)
    pipe = Pipeline([("pre", _make_pre()), ("rf", model)])
    pipe.fit(X, y_util)
    return pipe

def ajustar_pipeline(df_train: pd.DataFrame, optimized: bool = False):
    """
    Si optimized=True realiza búsqueda en rejilla (pequeña) con GroupKFold y NDCG@5.
    """
    X = df_train[NUM+CAT]
    y_rel = df_train["rel"].values
    qids = df_train["consulta_id"].values

    if optimized:
        # Intentamos LGBM Ranker con pequeña rejilla
        try:
            import lightgbm as lgb  # noqa: F401
            grid = [
                {"n_estimators": 300, "learning_rate": 0.08, "num_leaves": 63, "subsample":0.9, "colsample_bytree":0.9},
                {"n_estimators": 500, "learning_rate": 0.06, "num_leaves": 95, "subsample":0.9, "colsample_bytree":0.9},
                {"n_estimators": 400, "learning_rate": 0.10, "num_leaves": 63, "subsample":0.8, "colsample_bytree":0.8},
            ]
            gkf = GroupKFold(n_splits=5)
            best_score, best_params, best_pipe = -1.0, None, None
            for params in grid:
                fold_scores=[]
                for tr, te in gkf.split(X, groups=qids, y=y_rel):
                    pipe = _fit_lgbm_ranker_with_groups(X.iloc[tr], y_rel[tr],
                                                        df_train.iloc[tr]["consulta_id"].values, params)
                    score = eval_ndcg5_over_groups(pipe, X.iloc[te], y_rel[te], df_train.iloc[te]["consulta_id"].values)
                    fold_scores.append(score)
                mean_score = float(np.mean(fold_scores))
                if mean_score > best_score:
                    best_score, best_params = mean_score, params
            # Entrenamiento final con mejores params
            final_pipe = _fit_lgbm_ranker_with_groups(X, y_rel, qids, best_params)
            return final_pipe, {"modelo":"lightgbm_ranker", "ndcg5_cv": round(best_score,4), "params": best_params}
        except Exception:
            pass

        # Fallback: RandomForestRegressor optimizado (scoring NDCG@5 usando y_rel)
        y_util = df_train["utilidad"].values
        grid_rf = [
            {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
            {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1},
            {"n_estimators": 400, "max_depth": 20,   "min_samples_leaf": 2},
        ]
        gkf = GroupKFold(n_splits=5)
        best_score, best_params = -1.0, None
        for params in grid_rf:
            fold_scores=[]
            for tr, te in gkf.split(X, groups=qids, y=y_util):
                pipe = _fit_rf_regressor(X.iloc[tr], y_util[tr], params)
                # evaluamos NDCG@5 con etiquetas binarias "rel"
                score = eval_ndcg5_over_groups(pipe, X.iloc[te], y_rel[te], df_train.iloc[te]["consulta_id"].values)
                fold_scores.append(score)
            mean_score = float(np.mean(fold_scores))
            if mean_score > best_score:
                best_score, best_params = mean_score, params
        final_pipe = _fit_rf_regressor(X, y_util, best_params)
        return final_pipe, {"modelo":"random_forest_regression_fallback", "ndcg5_cv": round(best_score,4), "params": best_params}

    # --- modo rápido (sin grid) como antes ---
    try:
        import lightgbm as lgb  # noqa: F401
        params = {"n_estimators": 350, "learning_rate": 0.08, "num_leaves": 63, "subsample":0.9, "colsample_bytree":0.9}
        final_pipe = _fit_lgbm_ranker_with_groups(X, y_rel, qids, params)
        return final_pipe, {"modelo":"lightgbm_ranker", "params": params}
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        params = {"n_estimators": 350}
        y_util = df_train["utilidad"].values
        final_pipe = _fit_rf_regressor(X, y_util, params)
        return final_pipe, {"modelo":"random_forest_regression_fallback", "params": params}
