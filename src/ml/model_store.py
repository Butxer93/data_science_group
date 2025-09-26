from pathlib import Path
import os, joblib

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
PIPE_PATH = MODELS_DIR / "rank_refuel_pipe.joblib"

def load_rank_pipeline():
    return joblib.load(PIPE_PATH) if PIPE_PATH.exists() else None

def save_rank_pipeline(pipe):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, PIPE_PATH)

def env_weights(priority: str):
    def getf(name, default):
        try: return float(os.getenv(name, default))
        except: return default
    if priority == "sustainability":
        return {"w_saving":1.0, "w_detour":1.0, "w_wait":1.0, "w_co2":1.0, "shadow_price_co2": getf("FLEET_SHADOW_CO2", 0.2)}
    if priority == "cost":
        return {"w_saving":1.0, "w_detour":1.0, "w_wait":1.0, "w_co2":0.2, "shadow_price_co2": getf("FLEET_SHADOW_CO2", 0.0)}
    return {"w_saving":1.0, "w_detour":1.0, "w_wait":1.0, "w_co2":0.5, "shadow_price_co2": getf("FLEET_SHADOW_CO2", 0.1)}
