from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd, io, json
from pathlib import Path

from reco.utils import polyline_stub, distance_along, distance_to_polyline_km
from reco.ml.rank_refuel import rank_candidates
from reco.ml.ml_utils import load_rank_pipeline, save_rank_pipeline, env_weights
from reco.ml.train_ranker import build_synthetic_training, fit_pipeline

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

app = FastAPI(title="Desafio_R3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def load_csv(name): 
    p = DATA_DIR / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def get_vehicle(vehicle_id: str):
    vdf = load_csv("vehicles.csv")
    row = vdf[vdf["vehicle_id"] == vehicle_id]
    if row.empty:
        raise HTTPException(404, "vehicle not found")
    return row.iloc[0].to_dict()

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": load_rank_pipeline() is not None}

@app.post("/ml/train_ranker")
def ml_train_ranker(synthetic: bool = True):
    sdf = load_csv("stations.csv")
    if sdf.empty: raise HTTPException(400, "stations.csv not found")
    df = build_synthetic_training(sdf)
    pipe, meta = fit_pipeline(df)
    save_rank_pipeline(pipe)
    return {"ok": True, "meta": meta, "n_samples": len(df)}

@app.post("/ml/rank_refuel_by_route")
async def rank_refuel_by_route(req: Request):
    body = await req.json()
    route = body["route"]
    liters = float(body.get("liters_needed", 40.0))
    price_area_mean = float(body.get("price_area_mean", 1.62))
    priority = body.get("priority", "balanced")

    vehicle = get_vehicle(route["vehicle_id"])
    poly = polyline_stub((route["start_lat"], route["start_lon"]),
                         (route["end_lat"], route["end_lon"]), 40)

    candidates = []
    sdf = load_csv("stations.csv")
    for _, s in sdf.iterrows():
        if s["fuel_type"] != vehicle["fuel_type"]:
            continue
        detour = distance_to_polyline_km((s["lat"], s["lon"]), poly)
        if detour > 10: continue
        st = s.to_dict()
        st["detour_km"] = detour
        st["vehicle_type"] = vehicle["vehicle_type"]
        st["co2_stop_kg"] = liters * vehicle["emissions_factor_g_per_l"] / 1000.0
        candidates.append(st)

    if not candidates:
        return {"stations": [], "policy": priority}

    ctx = {
        "liters_needed": liters,
        "price_area_mean": price_area_mean,
        "eur_per_km": 0.18,
        "beta_wait": 0.05,
        "vehicle_type": vehicle["vehicle_type"],
    }
    ctx.update(env_weights(priority))

    ranked = rank_candidates(candidates, ctx, pipe=load_rank_pipeline())
    return {"stations": ranked, "policy": priority}
