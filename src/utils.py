import math
import pandas as pd

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def distancia_segmento_km(p, a, b):
    latp, lonp = p; lata, lona = a; latb, lonb = b
    x = (lonp - lona) * math.cos(math.radians((lata+latb)/2)) * 111.32
    y = (latp - lata) * 110.57
    dx = (lonb - lona) * math.cos(math.radians((lata+latb)/2)) * 111.32
    dy = (latb - lata) * 110.57
    if dx == 0 and dy == 0:
        return math.hypot(x, y)
    t = max(0.0, min(1.0, (x*dx + y*dy)/(dx*dx + dy*dy)))
    projx, projy = t*dx, t*dy
    return math.hypot(x - projx, y - projy)

def distancia_a_polilinea_km(pt, poli):
    best = 1e9
    for i in range(1, len(poli)):
        best = min(best, distancia_segmento_km(pt, poli[i-1], poli[i]))
    return best

def generar_polilinea_desde_rutas(ruta_id: str, df_rutas: pd.DataFrame):
    sub = df_rutas[df_rutas["ruta_id"] == ruta_id].copy()
    if sub.empty:
        return []
    # Ordenar por km_desde_origen si existe, si no, por índice
    if "km_desde_origen" in sub.columns:
        sub = sub.sort_values("km_desde_origen")
    return list(zip(sub["latitud"].astype(float), sub["longitud"].astype(float)))
