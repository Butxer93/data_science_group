from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd, io, json


from src.utils import polyline_stub, distance_along, distance_to_polyline_km
from src.ml.rank_refuel import rank_candidates
from src.ml.model_store import load_rank_pipeline, save_rank_pipeline, env_weights
from src.ml.train_ranker import build_synthetic_training, fit_pipeline
#from src.ml.habits_efficiency import vehicle_efficiency


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

app = FastAPI(title="Desafio_R3 API (FastAPI sin clases)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

def load_csv(name: str) -> pd.DataFrame:
    p = DATA_DIR / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def get_vehicle(vehicle_id: str) -> dict:
    vdf = load_csv("vehicles.csv")
    row = vdf[vdf["vehicle_id"] == vehicle_id]
    if row.empty:
        raise HTTPException(404, "vehicle not found")
    return row.iloc[0].to_dict()

def co2_stop_kg(vehicle: dict, liters: float) -> float:
    ef = float(vehicle.get("emissions_factor_g_per_l", 0.0))
    return liters * ef / 1000.0

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": load_rank_pipeline() is not None}

@app.post("/route/plan")
async def route_plan(req: Request):
    body = await req.json()
    route = body.get("route") or body
    poly = polyline_stub((float(route["start_lat"]), float(route["start_lon"])),
                         (float(route["end_lat"]),  float(route["end_lon"])), steps=30)
    acc = distance_along(poly)
    return {"points": poly, "distances_km": acc, "avoid_tolls": bool(route.get("avoid_tolls", False))}

@app.post("/ml/train_ranker")
def ml_train_ranker(synthetic: bool = True):
    sdf = load_csv("stations.csv")
    if sdf.empty:
        raise HTTPException(400, "stations.csv not found")
    df = build_synthetic_training(sdf, n_queries=200, cand_per_q=5) if synthetic else build_synthetic_training(sdf)
    pipe, meta = fit_pipeline(df)   # LGBM Ranker si está disponible; si no, RF fallback
    save_rank_pipeline(pipe)
    return {"ok": True, "meta": meta, "n_samples": int(len(df))}

@app.post("/ml/rank_refuel_by_route")
async def rank_refuel_by_route(req: Request):
    body = await req.json()
    route = body["route"]
    liters = float(body.get("liters_needed", 40.0))
    price_area_mean = float(body.get("price_area_mean", 1.62))
    priority = body.get("priority", "balanced")

    vehicle = get_vehicle(route["vehicle_id"])
    poly = polyline_stub((route["start_lat"], route["start_lon"]),
                         (route["end_lat"],  route["end_lon"]), steps=40)

    candidates = []
    user_cands = body.get("candidates_fuel")
    if user_cands:
        for c in user_cands:
            detour = distance_to_polyline_km((c["lat"], c["lon"]), poly)
            st = dict(c)
            st["detour_km"] = detour
            st["vehicle_type"] = vehicle.get("vehicle_type", "unknown")
            st["co2_stop_kg"] = 0.0 if vehicle.get("fuel_type") == "ev" else co2_stop_kg(vehicle, liters)
            candidates.append(st)
    else:
        sdf = load_csv("stations.csv")
        for _, s in sdf.iterrows():
            if vehicle.get("fuel_type") not in (None, "ev") and s["fuel_type"] != vehicle["fuel_type"]:
                continue
            detour = distance_to_polyline_km((s["lat"], s["lon"]), poly)
            if detour <= 10.0:
                st = s.to_dict()
                st["detour_km"] = detour
                st["vehicle_type"] = vehicle.get("vehicle_type", "unknown")
                st["co2_stop_kg"] = 0.0 if vehicle.get("fuel_type") == "ev" else co2_stop_kg(vehicle, liters)
                candidates.append(st)

    if not candidates:
        return {"stations": [], "policy": priority}

    ctx = {
        "liters_needed": liters,
        "price_area_mean": price_area_mean,
        "eur_per_km": 0.18,
        "beta_wait": 0.05,
        "vehicle_type": vehicle.get("vehicle_type", "unknown"),
    }
    ctx.update(env_weights(priority))  # pesos por política desde ENV

    pipe = load_rank_pipeline()  # puede ser None (baseline puro)
    ranked = rank_candidates(candidates, ctx, pipe=pipe)
    return {"stations": ranked, "policy": priority, "used_model": bool(pipe)}

# ===== Ingesta EV opcional (sin librerías de frontend) =====
@app.post("/ingest/ev/openchargemap")
async def ingest_openchargemap(file: UploadFile = File(...)):
    try:
        body = await file.read()
        data = json.loads(body.decode("utf-8"))
        src = data if isinstance(data, list) else data.get("data", [])
        rows = []
        for item in src:
            addr = item.get("AddressInfo", {}) or {}
            lat, lon, name = addr.get("Latitude"), addr.get("Longitude"), addr.get("Title")
            plugs = []
            for c in item.get("Connections", []) or []:
                ct = (c.get("ConnectionType") or {}).get("Title")
                if ct: plugs.append(ct)
            max_kw = max([c.get("PowerKW") or 0 for c in item.get("Connections", []) or []], default=0)
            rows.append({
                "charge_id": item.get("ID", ""), "name": name, "lat": lat, "lon": lon,
                "network": (item.get("OperatorInfo") or {}).get("Title"),
                "plug_types": ",".join(sorted(set(plugs))), "max_power_kw": max_kw,
                "price_eur_per_kwh": None, "wait_min": None
            })
        df = pd.DataFrame(rows)
        (DATA_DIR / "ev_stations.csv").write_text(df.to_csv(index=False))
        return {"ok": True, "rows": int(len(df))}
    except Exception as e:
        raise HTTPException(400, f"Invalid JSON: {e}")

@app.post("/ingest/ev/datos_gob")
async def ingest_datos_gob(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        colmap = {"Identificador":"charge_id","Nombre":"name","Latitud":"lat","Longitud":"lon",
                  "Operador":"network","TiposConector":"plug_types","PotenciaKW":"max_power_kw","PrecioKWh":"price_eur_per_kwh"}
        for src, dst in colmap.items():
            if src in df.columns: df[dst] = df[src]
        keep = [c for c in ["charge_id","name","lat","lon","network","plug_types","max_power_kw","price_eur_per_kwh"] if c in df.columns]
        out_df = df[keep].copy(); out_df["wait_min"] = None
        (DATA_DIR / "ev_stations.csv").write_text(out_df.to_csv(index=False))
        return {"ok": True, "rows": int(len(out_df))}
    except Exception as e:
        raise HTTPException(400, f"Invalid CSV: {e}")
