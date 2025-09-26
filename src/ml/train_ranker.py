import pandas as pd, numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

NUM = ["delta_price","detour_km","wait_min","liters_needed"]
CAT = ["brand","fuel_type","vehicle_type"]

def build_synthetic_training(df_stations: pd.DataFrame, n_queries=200, cand_per_q=5):
    rng = np.random.default_rng(42)
    rows=[]
    brands = df_stations["brand"].dropna().unique().tolist() or ["BrandA","BrandB","BrandC"]
    fuels  = df_stations["fuel_type"].dropna().unique().tolist() or ["diesel","gasoline"]
    for q in range(n_queries):
        vehicle_type = rng.choice(["car","van","truck"], p=[0.55,0.35,0.10])
        price_area_mean = float(np.clip(rng.normal(1.62, 0.04), 1.45, 1.80))
        liters_needed = float(np.clip(rng.normal(42, 8), 12, 70))
        for _ in range(cand_per_q):
            brand = rng.choice(brands)
            fuel  = rng.choice(fuels)
            price = float(np.clip(price_area_mean + rng.normal(0,0.03), 1.40, 1.85))
            detour= float(np.clip(abs(rng.normal(1.0, 1.0)), 0, 10))
            wait  = float(np.clip(rng.normal(3,2), 0, 20))

            saving = (price_area_mean - price) * liters_needed
            penalty = detour*0.18 + 0.05*wait
            utility = saving - penalty + (0.05 if brand=="BrandB" else 0.0)

            rows.append(dict(
                query_id=q, brand=brand, fuel_type=fuel, vehicle_type=vehicle_type,
                price_per_liter=price, wait_min=wait, detour_km=detour,
                liters_needed=liters_needed, price_area_mean=price_area_mean,
                delta_price=price_area_mean - price, utility=utility
            ))
    df = pd.DataFrame(rows)
    # relevancia binaria (mejor por utilidad)
    df["rel"] = 0
    df.loc[df.groupby("query_id")["utility"].idxmax(), "rel"] = 1
    return df

def fit_pipeline(df_train: pd.DataFrame):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)
    ])
    # Intentamos LGBM Ranker (LambdaRank). Si no está disponible, fallback a RF
    try:
        import lightgbm as lgb
        model = lgb.LGBMRanker(objective="lambdarank", n_estimators=300,
                               learning_rate=0.08, num_leaves=63,
                               subsample=0.9, colsample_bytree=0.9, random_state=42)
        pipe = Pipeline([("pre", pre), ("ranker", model)])
        groups = df_train.groupby("query_id").size().to_list()
        X = df_train[NUM+CAT]; y = df_train["rel"]
        pipe.fit(X, y, ranker__group=groups)
        return pipe, {"used":"lightgbm"}
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        pipe = Pipeline([("pre", pre), ("rf", model)])
        X = df_train[NUM+CAT]; y = df_train["utility"]
        pipe.fit(X, y)
        return pipe, {"used":"random_forest_regression_fallback"}
