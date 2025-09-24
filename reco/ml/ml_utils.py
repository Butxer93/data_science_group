from pathlib import Path
import os, joblib

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
PIPE_PATH = MODELS_DIR / "rank_refuel_pipe.joblib"

def load_rank_pipeline():
    return joblib.load(PIPE_PATH) if PIPE_PATH.exists() else None

def save_rank_pipeline(pipe):
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, PIPE_PATH)

def env_weights(priority):
    def g(name, default): 
        try: return float(os.getenv(name,default))
        except: return default
    if priority=="sustainability":
        return {"w_saving":1,"w_detour":1,"w_wait":1,"w_co2":1,"shadow_price_co2":0.2}
    if priority=="cost":
        return {"w_saving":1,"w_detour":1,"w_wait":1,"w_co2":0.2,"shadow_price_co2":0}
    return {"w_saving":1,"w_detour":1,"w_wait":1,"w_co2":0.5,"shadow_price_co2":0.1}
