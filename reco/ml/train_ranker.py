import pandas as pd, numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_synthetic_training(df_stations, n_queries=50, cand_per_q=5):
    rng = np.random.default_rng(42)
    rows=[]
    brands=df_stations["brand"].unique()
    fuels=df_stations["fuel_type"].unique()
    for q in range(n_queries):
        liters = float(np.clip(rng.normal(40,10),10,70))
        mean_price = float(np.clip(rng.normal(1.62,0.05),1.4,1.8))
        for i in range(cand_per_q):
            brand=rng.choice(brands); fuel=rng.choice(fuels)
            price=mean_price+rng.normal(0,0.03)
            detour=abs(rng.normal(2,1))
            wait=max(0,rng.normal(3,2))
            util=(mean_price-price)*liters - detour*0.2 - wait*0.05
            rows.append(dict(query=q,brand=brand,fuel_type=fuel,
                             price_per_liter=price,detour_km=detour,
                             wait_min=wait,liters_needed=liters,
                             delta_price=mean_price-price,
                             vehicle_type="car",rel=0 if util<0 else 1,utility=util))
    return pd.DataFrame(rows)

def fit_pipeline(df):
    NUM=["delta_price","detour_km","wait_min","liters_needed"]
    CAT=["brand","fuel_type","vehicle_type"]
    pre=ColumnTransformer([("num",StandardScaler(),NUM),("cat",OneHotEncoder(),CAT)])
    try:
        import lightgbm as lgb
        model=lgb.LGBMRanker(objective="lambdarank")
        pipe=Pipeline([("pre",pre),("model",model)])
        groups=df.groupby("query").size().to_list()
        X,y=df[NUM+CAT],df["rel"]
        pipe.fit(X,y,model__group=groups)
        return pipe,{"used":"lightgbm"}
    except:
        from sklearn.ensemble import RandomForestRegressor
        model=RandomForestRegressor()
        pipe=Pipeline([("pre",pre),("rf",model)])
        pipe.fit(df[NUM+CAT],df["utility"])
        return pipe,{"used":"rf-regression"}
