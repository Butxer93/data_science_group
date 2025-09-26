# scripts/generar_datos.py
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

def gen_puntos_repostaje(n=350):
    marcas = ["Repsol", "Cepsa", "Shell", "Galp", "BP", "Petronor"]
    carreteras = ["A-1","A-2","A-3","A-4","A-5","AP-2","AP-7"]
    servicios_pool = ["24H","WC","Tienda","Restaurante","Aire","Agua","Autolavado"]
    base_lat, base_lon = 40.4168, -3.7038

    rows=[]
    for i in range(n):
        marca = rng.choice(marcas)
        nombre = f"Estación {marca} {i:03d}"
        lat = base_lat + rng.normal(0, 1.5)/10
        lon = base_lon + rng.normal(0, 1.5)/10
        carr = rng.choice(carreteras)
        servicios = "|".join(rng.choice(servicios_pool, size=rng.integers(1,4), replace=False))
        precio = float(np.clip(1.55 + rng.normal(0, 0.05), 1.40, 1.90))
        espera = int(np.clip(rng.normal(4, 3), 0, 20))
        fuel = rng.choice(["diesel","gasolina"], p=[0.6,0.4])

        rows.append(dict(
            punto_id=f"P{i+1:04d}",
            nombre_estacion=nombre,
            marca=marca,
            latitud=round(lat, 5),
            longitud=round(lon, 5),
            carretera=carr,
            direccion=f"Área {carr} km {rng.integers(5, 600)}",
            servicios=servicios,
            precio_litro=round(precio, 3),
            minutos_espera=int(espera),
            tipo_combustible=fuel
        ))
    df = pd.DataFrame(rows)
    (DATA / "puntos_repostaje.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    print(f"[ok] puntos_repostaje.csv -> {len(df)} filas")

def gen_rutas(n_rutas=5, puntos_por_ruta=60):
    # Creamos varias rutas con puntos a lo largo del camino (lat/lon “lineales” + ruido)
    base_pairs = [
        ((40.4168,-3.7038),(41.3874,2.1686),"Madrid-Barcelona"),
        ((40.4168,-3.7038),(39.4699,-0.3763),"Madrid-Valencia"),
        ((40.4168,-3.7038),(37.3891,-5.9845),"Madrid-Sevilla"),
        ((40.4168,-3.7038),(43.2630,-2.9350),"Madrid-Bilbao"),
        ((40.4168,-3.7038),(36.7213,-4.4214),"Madrid-Málaga"),
    ]
    rows=[]
    for idx in range(n_rutas):
        (lat1,lon1), (lat2,lon2), desc = base_pairs[idx % len(base_pairs)]
        for k in range(puntos_por_ruta):
            t = k/(puntos_por_ruta-1)
            lat = lat1 + (lat2-lat1)*t + rng.normal(0, 0.01)
            lon = lon1 + (lon2-lon1)*t + rng.normal(0, 0.01)
            km = int(t * rng.integers(300, 700))
            carr = rng.choice(["A-1","A-2","A-3","AP-2","AP-7"])
            rows.append(dict(
                ruta_id=f"R{idx+1:03d}",
                descripcion=desc,
                latitud=round(lat, 5),
                longitud=round(lon, 5),
                carretera=carr,
                km_desde_origen=km
            ))
    df = pd.DataFrame(rows)
    df = df.sort_values(["ruta_id","km_desde_origen"])
    (DATA / "rutas.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    print(f"[ok] rutas.csv -> {len(df)} filas ({df['ruta_id'].nunique()} rutas)")

def gen_puntos_recarga(n=220):
    operadores = ["Iberdrola","Endesa","Zunder","Wenea","Tesla","Ionnity"]
    conectores = ["CCS2","Type2","CHAdeMO"]
    carreteras = ["A-1","A-2","AP-2","AP-7","A-3","A-5"]
    base_lat, base_lon = 40.4168, -3.7038
    rows=[]
    for i in range(n):
        op = rng.choice(operadores)
        nombre = f"{op} Punto {i:03d}"
        lat = base_lat + rng.normal(0, 1.5)/10
        lon = base_lon + rng.normal(0, 1.5)/10
        carr = rng.choice(carreteras)
        pw = float(np.clip(rng.normal(75, 40), 7, 350))
        precio = float(np.clip(rng.normal(0.42, 0.06), 0.25, 0.75))
        rows.append(dict(
            punto_ev_id=f"E{i+1:04d}",
            nombre_punto=nombre,
            operador=op,
            latitud=round(lat, 5),
            longitud=round(lon, 5),
            carretera=carr,
            direccion=f"Área {carr} km {rng.integers(5, 600)}",
            tipo_conector=rng.choice(conectores, p=[0.6,0.3,0.1]),
            potencia_kw=round(pw, 1),
            precio_kwh=round(precio, 3),
            disponible_24h=rng.choice(["si","no"], p=[0.7,0.3])
        ))
    df = pd.DataFrame(rows)
    (DATA / "puntos_recarga_electrica.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    print(f"[ok] puntos_recarga_electrica.csv -> {len(df)} filas")

def gen_beneficios_tarjeta(n_tarjetas=6, pct_global=True):
    # Algunas tarjetas por céntimos/litro, otras por porcentaje; y listas de estaciones
    marcas = ["Repsol","Cepsa","Shell","Galp","BP","Petronor"]
    rows=[]
    for i in range(n_tarjetas):
        nombre = ["MiTarjeta","SuperAhorrador","MaxDescuento","ProFleet","EcoPlus","RutaPro"][i % 6]
        tipo = rng.choice(["centimos","porcentaje"])
        cent = rng.integers(1, 8) if tipo == "centimos" else ""
        pct = round(float(rng.choice([1.0,1.5,2.0,2.5,3.0])), 1) if tipo == "porcentaje" else ""
        if rng.random() < 0.5:
            # aplica a todas
            estaciones = ""
        else:
            estaciones = "|".join([f"Estación {rng.choice(marcas)} {rng.integers(0, 200):03d}" for _ in range(rng.integers(1,4))])
        rows.append(dict(
            tarjeta=nombre,
            centimos_litro=cent,
            porcentaje=pct,
            estaciones_incluidas=estaciones
        ))
    df = pd.DataFrame(rows)
    (DATA / "beneficios_tarjeta.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    print(f"[ok] beneficios_tarjeta.csv -> {len(df)} filas")

if __name__ == "__main__":
    gen_puntos_repostaje(350)
    gen_rutas(5, 70)
    gen_puntos_recarga(220)
    gen_beneficios_tarjeta(6)
    print("[done] Datos sintéticos generados en ./data")
