import pandas as pd, numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

NUM = ["delta_precio","desvio_km","minutos_espera","litros_necesarios"]
CAT = ["marca","tipo_combustible","carretera"]

def _sintetizar_campos_faltantes(puntos: pd.DataFrame) -> pd.DataFrame:
    df = puntos.copy()
    if "precio_litro" not in df.columns:
        df["precio_litro"] = (1.50 + (df.index % 7) * 0.01).clip(1.35, 1.95)
    if "minutos_espera" not in df.columns:
        df["minutos_espera"] = (df.index % 10) + 1
    if "tipo_combustible" not in df.columns:
        df["tipo_combustible"] = ["diesel" if i % 2 == 0 else "gasolina" for i in range(len(df))]
    return df

def construir_entrenamiento_sintetico(puntos: pd.DataFrame, n_consultas=250, cand_por_q=6, seed=42):
    rng = np.random.default_rng(seed)
    puntos = _sintetizar_campos_faltantes(puntos)
    rows=[]
    marcas = puntos["marca"].dropna().unique().tolist() or ["BrandA","BrandB"]
    fuels  = puntos["tipo_combustible"].dropna().unique().tolist() or ["diesel","gasolina"]
    carrets= puntos["carretera"].dropna().unique().tolist() or ["A-1","A-2"]
    for q in range(n_consultas):
        precio_area = float(np.clip(rng.normal(1.62, 0.04), 1.45, 1.95))
        litros = float(np.clip(rng.normal(42, 8), 12, 70))
        for _ in range(cand_por_q):
            marca = rng.choice(marcas); fuel = rng.choice(fuels); carr = rng.choice(carrets)
            precio = float(np.clip(precio_area + rng.normal(0,0.03), 1.40, 1.95))
            desvio = float(np.clip(abs(rng.normal(1.0, 1.0)), 0, 10))
            espera = float(np.clip(rng.normal(3,2), 0, 20))
            ahorro = max(0.0, precio_area - precio) * litros
            penal = desvio*0.18 + 0.05*espera + 0.1  # +0.1 por ruido
            utilidad = ahorro - penal + (0.05 if marca == "Repsol" else 0.0)
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

def ajustar_pipeline(df_train: pd.DataFrame):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)
    ])
    try:
        import lightgbm as lgb
        model = lgb.LGBMRanker(objective="lambdarank", n_estimators=350,
                               learning_rate=0.08, num_leaves=63,
                               subsample=0.9, colsample_bytree=0.9, random_state=42)
        pipe = Pipeline([("pre", pre), ("ranker", model)])
        X = df_train[NUM+CAT]; y = df_train["rel"].values
        grupos = df_train.groupby("consulta_id").size().to_list()
        pipe.fit(X, y, ranker__group=grupos)
        return pipe, {"modelo":"lightgbm_ranker","filas":int(len(df_train))}
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=350, random_state=42)
        pipe = Pipeline([("pre", pre), ("rf", model)])
        X = df_train[NUM+CAT]; y = df_train["utilidad"].values
        pipe.fit(X, y)
        return pipe, {"modelo":"random_forest_regression_fallback","filas":int(len(df_train))}
