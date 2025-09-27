# streamlit_app.py
# Dashboard auxiliar para cliente: Repostaje + Hábitos (API unificada)
# Ejecuta: uv run streamlit run streamlit_app.py

import os
import math
import io
from pathlib import Path
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Configuración inicial
# =========================
st.set_page_config(
    page_title="Cuadro de mando — Repostaje & Hábitos",
    page_icon="⛽️",
    layout="wide",
)

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "sinteticos"
MODELS_DIR = ROOT / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Helpers HTTP y utilidades
# =========================
def build_headers(token: str | None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"
    return headers

def call_post(url: str, params=None, payload=None, timeout=180, token: str | None = None):
    try:
        r = requests.post(url, params=params, json=payload, timeout=timeout, headers=build_headers(token))
        if r.status_code >= 400:
            return {"ok": False, "error": f"{r.status_code} {r.text}"}
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

def call_get(url: str, params=None, timeout=90, token: str | None = None):
    try:
        r = requests.get(url, params=params, timeout=timeout, headers=build_headers(token))
        if r.status_code >= 400:
            return {"ok": False, "error": f"{r.status_code} {r.text}"}
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

@st.cache_data(show_spinner=False)
def leer_csv_local(nombre: str) -> pd.DataFrame:
    p = DATA_DIR / nombre
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def kpi_delta(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:+.2f}"

def df_to_csv_download(df: pd.DataFrame, name: str):
    buff = io.StringIO()
    df.to_csv(buff, index=False)
    st.download_button(
        label=f"💾 Descargar {name}.csv",
        data=buff.getvalue(),
        file_name=f"{name}.csv",
        mime="text/csv",
        use_container_width=True,
    )

def save_csv_no_blank(df: pd.DataFrame, path: Path):
    df = df.dropna(how="all").copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")

# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ Configuración")
api = st.sidebar.text_input("Base URL API", API_BASE)
api_token = st.sidebar.text_input("Token (Bearer)", type="password", help="Si tu API requiere autenticación, inclúyelo aquí.")

if st.sidebar.button("Probar /health"):
    h = call_get(f"{api}/health", token=api_token)
    st.sidebar.write(h)

st.sidebar.divider()
st.sidebar.caption("Datos locales (./data/sinteticos)")
if st.sidebar.button("↻ Recargar caché CSV"):
    st.cache_data.clear()
    st.sidebar.success("Caché limpiada.")

# =========================
# Tabs
# =========================
tab_datos, tab_repostaje, tab_habitos = st.tabs([
    "🏗️ Datos & Entrenamiento",
    "⛽️ Recomendaciones de Repostaje",
    "🌱 Hábitos Eficientes",
])

# =============================================================================
# TAB 1 — Datos & Entrenamiento
# =============================================================================
with tab_datos:
    st.subheader("Generación de datos sintéticos")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Repostaje (CSV)**")
        n_puntos = st.number_input("Puntos de repostaje", 50, 100000, 360, 10, key="gen_puntos")
        n_rutas = st.number_input("Rutas", 1, 2000, 5, 1, key="gen_rutas")
        ppr = st.number_input("Puntos por ruta", 5, 5000, 70, 5, key="gen_ppr")
        n_ev = st.number_input("Puntos EV", 50, 200000, 220, 10, key="gen_ev")
        n_tarjetas = st.number_input("Tarjetas", 1, 5000, 6, 1, key="gen_tarjetas")
        if st.button("🧪 Generar CSV de Repostaje"):
            with st.spinner("Generando CSV de repostaje en ./data/sinteticos..."):
                res = call_post(
                    f"{api}/repostaje/data/generar_sintetico",
                    params=dict(
                        n_puntos=int(n_puntos),
                        n_rutas=int(n_rutas),
                        puntos_por_ruta=int(ppr),
                        n_ev=int(n_ev),
                        n_tarjetas=int(n_tarjetas),
                    ),
                    token=api_token,
                )
            if res.get("ok"):
                st.success("CSV generados correctamente.")
                st.json(res)
            else:
                st.error(res.get("error", "Error"))

    with c2:
        st.markdown("**Hábitos (CSV)**")
        hN = st.number_input("Filas telemetría", 500, 200000, 6000, 500, key="habN")
        hSeed = st.number_input("Seed", 0, 1_000_000, 123, 1, key="habSeed")
        if st.button("🌿 Generar CSV de Hábitos"):
            with st.spinner("Generando CSV de hábitos en ./data/sinteticos..."):
                res = call_post(
                    f"{api}/habitos/data/generar_sintetico",
                    params=dict(N=int(hN), seed=int(hSeed)),
                    token=api_token,
                )
            if res.get("ok"):
                st.success("CSV de hábitos generado.")
                st.json(res)
            else:
                st.error(res.get("error", "Error"))

    st.divider()
    st.subheader("Subir CSV propios (reemplazo de sintéticos)")

    expected_cols = {
        "puntos_repostaje.csv": [
            "punto_id","nombre_estacion","marca","latitud","longitud","carretera","direccion","servicios",
            "precio_litro","minutos_espera","tipo_combustible"
        ],
        "rutas.csv": [
            "ruta_id","descripcion","latitud","longitud","carretera","km_desde_origen"
        ],
        "puntos_recarga_electrica.csv": [
            "punto_ev_id","nombre_punto","operador","latitud","longitud","carretera","direccion",
            "tipo_conector","potencia_kw","precio_kwh","disponible_24h"
        ],
        "beneficios_tarjeta.csv": [
            "tarjeta","centimos_litro","porcentaje","estaciones_incluidas"
        ],
        "habitos_telemetria.csv": [
            "vehiculo_id","velocidad_media_kmh","frenadas_fuertes_100km","aceleraciones_100km","ratio_ralenti",
            "ratio_carga","consumo_l_100km","eficiente"
        ],
    }

    up1, up2 = st.columns(2)
    with up1:
        st.caption("⛽ Repostaje")
        for name in ["puntos_repostaje.csv","rutas.csv","puntos_recarga_electrica.csv","beneficios_tarjeta.csv"]:
            file = st.file_uploader(f"Subir {name}", type=["csv"], key=f"u_{name}")
            if file is not None:
                try:
                    df = pd.read_csv(file)
                    missing = [c for c in expected_cols[name] if c not in df.columns]
                    if missing:
                        st.error(f"{name}: faltan columnas {missing}")
                    else:
                        save_csv_no_blank(df, DATA_DIR / name)
                        st.success(f"{name} guardado en ./data/sinteticos/{name}")
                except Exception as e:
                    st.error(f"{name}: error leyendo CSV: {e}")

    with up2:
        st.caption("🌱 Hábitos")
        file = st.file_uploader("Subir habitos_telemetria.csv", type=["csv"], key="u_habitos")
        if file is not None:
            try:
                df = pd.read_csv(file)
                missing = [c for c in expected_cols["habitos_telemetria.csv"] if c not in df.columns]
                if missing:
                    st.error(f"habitos_telemetria.csv: faltan columnas {missing}")
                else:
                    save_csv_no_blank(df, DATA_DIR / "habitos_telemetria.csv")
                    st.success("habitos_telemetria.csv guardado en ./data/sinteticos/habitos_telemetria.csv")
            except Exception as e:
                st.error(f"habitos_telemetria.csv: error leyendo CSV: {e}")

    st.divider()
    st.subheader("Entrenamiento de modelos")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Ranker de Repostaje**")
        r_opt = st.toggle("Optimizado (NDCG@5, GKF + grid)", True, key="rank_opt")
        q = st.number_input("Consultas sintéticas", 50, 5000, 250, 50, key="rank_q")
        k = st.number_input("Candidatos por consulta", 2, 50, 6, 1, key="rank_k")
        if st.button("🏎️ Entrenar Ranker"):
            with st.spinner("Entrenando ranker..."):
                res = call_post(
                    f"{api}/repostaje/ml/entrenar_ranker",
                    params=dict(syntetico=True, optimized=bool(r_opt), n_consultas=int(q), candidatos_por_consulta=int(k)),
                    token=api_token,
                )
            if res.get("ok"):
                st.success("Ranker entrenado y guardado.")
                st.json(res)
            else:
                st.error(res.get("error", "Error"))

        info = call_get(f"{api}/repostaje/ml/info", token=api_token)
        if isinstance(info, dict):
            st.caption("Estado pipeline:")
            st.json(info)

    with c4:
        st.markdown("**Hábitos Eficientes**")
        h_opt = st.toggle("Optimizado (RF grid + KMeans silhouette)", True, key="hab_opt")
        reuse_ds = st.toggle("Reutilizar CSV si existe", True, key="hab_reuse")
        hN2 = st.number_input("Filas telemetría (si genera)", 500, 200000, 6000, 500, key="habN2")
        hSeed2 = st.number_input("Seed (si genera)", 0, 1_000_000, 123, 1, key="habSeed2")
        if st.button("🌱 Entrenar Hábitos"):
            with st.spinner("Entrenando hábitos..."):
                res = call_post(
                    f"{api}/habitos/ml/entrenar",
                    params=dict(
                        optimized=bool(h_opt),
                        reuse_if_exists=bool(reuse_ds),
                        N=int(hN2),
                        seed=int(hSeed2),
                    ),
                    token=api_token,
                )
            if isinstance(res, dict) and res.get("ok"):
                st.success("Modelos de hábitos entrenados.")
                st.json(res)
            else:
                st.error(res.get("error", "Error"))

    st.divider()
    st.subheader("Vista rápida de CSV locales (./data/sinteticos)")
    cols = st.columns(4)
    files = [
        ("puntos_repostaje.csv", cols[0]),
        ("rutas.csv", cols[1]),
        ("puntos_recarga_electrica.csv", cols[2]),
        ("beneficios_tarjeta.csv", cols[3]),
    ]
    for fname, col in files:
        with col:
            df = leer_csv_local(fname)
            st.caption(f"**{fname}** — {len(df)} filas")
            if not df.empty:
                st.dataframe(df.head(12), use_container_width=True, height=250)
                df_to_csv_download(df.head(200), fname.replace(".csv", "_preview"))
            else:
                st.info("No encontrado o vacío.")

    st.divider()
    st.subheader("Dataset de Hábitos — preview (API)")
    prev = call_get(f"{api}/habitos/data/telemetria", params={"limit": 10}, token=api_token)
    if prev and prev.get("ok"):
        st.write(f"Filas: {prev.get('rows')}")
        st.dataframe(pd.DataFrame(prev.get("preview", [])), use_container_width=True, height=260)
    else:
        st.info(prev.get("error", "No existe todavía. Usa 'Generar CSV de Hábitos'"))
        st.caption("Ruta esperada: ./data/sinteticos/habitos_telemetria.csv")

# =============================================================================
# TAB 2 — Recomendaciones de Repostaje
# =============================================================================
with tab_repostaje:
    st.subheader("Ranking de estaciones en ruta")

    # Rutas disponibles vía CSV local o endpoint
    rutas_df = leer_csv_local("rutas.csv")
    rutas_ids = sorted(rutas_df["ruta_id"].astype(str).str.strip().unique().tolist()) if not rutas_df.empty else []
    if not rutas_ids:
        # fallback: pedir a la API
        rutas_api = call_get(f"{api}/repostaje/data/rutas", token=api_token)
        if isinstance(rutas_api, dict) and "ruta_ids" in rutas_api:
            rutas_ids = rutas_api["ruta_ids"]

    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1.2])
    with c1:
        ruta_id = st.selectbox("Ruta (ruta_id)", rutas_ids, index=0 if rutas_ids else None)
    with c2:
        litros = st.number_input("Litros necesarios", 5.0, 120.0, 45.0, 1.0)
    with c3:
        precio_area = st.number_input("Precio medio zona (€/L)", 0.5, 3.5, 1.62, 0.01)
    with c4:
        prioridad = st.selectbox("Prioridad", ["equilibrado", "coste", "sostenible"], index=0)

    c5, c6, c7 = st.columns([1, 1, 1])
    with c5:
        tipo_cliente = st.selectbox("Tipo de cliente", ["particular", "flota"])
    with c6:
        tarjeta = st.text_input("Tarjeta (opcional)", value="MiTarjeta")
    with c7:
        pedir = st.button("🔎 Obtener ranking")

    if pedir and ruta_id:
        payload = dict(
            ruta_id=str(ruta_id),
            litros_necesarios=float(litros),
            precio_area_medio=float(precio_area),
            prioridad=prioridad,
            tipo_cliente=tipo_cliente,
            tarjeta=(tarjeta or None),
        )
        with st.spinner("Consultando recomendaciones..."):
            res = call_post(f"{api}/repostaje/recomendaciones", payload=payload, token=api_token)

        if isinstance(res, dict) and res.get("estaciones") is not None:
            estaciones = pd.DataFrame(res["estaciones"])
            if estaciones.empty:
                st.warning("No hay candidatos (quizá no hay estaciones a ≤10km de la ruta).")
            else:
                top = estaciones.sort_values("score", ascending=False).reset_index(drop=True)
                top1 = top.iloc[0]

                precio_litro = float(top1.get("precio_litro", precio_area))
                precio_neto = float(top1.get("precio_neto", precio_litro))
                delta = (precio_area - precio_neto)
                ahorro = max(0.0, delta) * float(litros)
                desvio = float(top1.get("desvio_km", 0))
                espera = float(top1.get("minutos_espera", 0))
                co2 = 2.64 * float(litros)  # simple

                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("Ahorro estimado TOP1 (€)", f"{ahorro:.2f}", kpi_delta(delta))
                with k2:
                    st.metric("Precio neto TOP1 (€/L)", f"{precio_neto:.3f}", kpi_delta(precio_area - precio_litro))
                with k3:
                    st.metric("Desvío TOP1 (km)", f"{desvio:.2f}")
                with k4:
                    st.metric("CO₂ parada (kg)", f"{co2:.1f}")

                st.markdown("#### Estaciones (Top 25)")
                show_cols = [
                    "punto_id","nombre_estacion","marca","carretera","precio_litro","precio_neto",
                    "minutos_espera","desvio_km","tipo_combustible","score","latitud","longitud"
                ]
                for c in show_cols:
                    if c not in top.columns:
                        top[c] = np.nan
                st.dataframe(top[show_cols].head(25), use_container_width=True, height=520)

                # Mapa rápido
                if {"latitud","longitud"}.issubset(top.columns):
                    try:
                        map_df = top.rename(columns={"latitud":"lat", "longitud":"lon"})[["lat","lon"]].head(200)
                        st.map(map_df)
                    except Exception:
                        pass

                # Export CSV
                st.markdown("#### Exportar")
                df_to_csv_download(top.head(100), "ranking_repostaje_top")

                st.caption(
                    f"Modelo usado: {'✅' if res.get('modelo_usado') else 'heurística'} · "
                    f"Prioridad: {res.get('prioridad')} · Candidatos: {res.get('n_candidatos')}"
                )
        else:
            st.error(res.get("error", "Error consultando ranking") if isinstance(res, dict) else "Error desconocido")

# =============================================================================
# TAB 3 — Hábitos Eficientes
# =============================================================================
with tab_habitos:
    st.subheader("Predicción y consejos de eco-driving")

    c1, c2, c3 = st.columns(3)
    with c1:
        vel = st.number_input("Velocidad media (km/h)", 10.0, 150.0, 72.0, 1.0)
        fren = st.number_input("Frenadas fuertes (por 100km)", 0.0, 30.0, 4.0, 1.0)
    with c2:
        acel = st.number_input("Aceleraciones (por 100km)", 0.0, 30.0, 6.0, 1.0)
        ral = st.number_input("Ratio de ralentí", 0.0, 1.0, 0.10, 0.01)
    with c3:
        carga = st.number_input("Ratio de carga (0–1)", 0.0, 1.0, 0.50, 0.05)
        pedir_hab = st.button("🌿 Predecir hábitos")

    if pedir_hab:
        payload = {
            "velocidad_media_kmh": float(vel),
            "frenadas_fuertes_100km": float(fren),
            "aceleraciones_100km": float(acel),
            "ratio_ralenti": float(ral),
            "ratio_carga": float(carga),
        }
        with st.spinner("Consultando modelo de hábitos..."):
            res = call_post(f"{api}/habitos/predict", payload=payload, token=api_token)
        if isinstance(res, dict) and "eficiente" in res:
            ca, cb, cc = st.columns(3)
            with ca:
                st.metric("¿Eficiente?", "Sí" if res["eficiente"]==1 else "No")
            with cb:
                prob = res.get("prob_eficiente")
                st.metric("Prob. eficiente", f"{prob:.2f}" if prob is not None else "—")
            with cc:
                st.metric("Cluster", str(res.get("cluster", "—")))

            st.markdown("#### Consejos sugeridos")
            tips = res.get("consejos") or []
            if tips:
                st.write("\n".join([f"- {t}" for t in tips]))
            else:
                st.info("Sin consejos específicos para este perfil.")
        else:
            st.error(res.get("error", "Error"))

    st.divider()
    st.subheader("Estado del servicio de hábitos")
    health = call_get(f"{api}/health", token=api_token)
    if isinstance(health, dict):
        st.json(health.get("habitos", health))
    else:
        st.error("No se pudo consultar /health")

    st.divider()
    st.subheader("Dataset de Hábitos (API)")
    colA, colB = st.columns([1, 1])
    with colA:
        prev = call_get(f"{api}/habitos/data/telemetria", params={"limit": 12}, token=api_token)
        if prev and prev.get("ok"):
            st.caption(f"Filas: {prev.get('rows')}")
            st.dataframe(pd.DataFrame(prev.get("preview", [])), use_container_width=True, height=280)
        else:
            st.info(prev.get("error", "Dataset no existente"))
    with colB:
        st.caption("Regenerar CSV de hábitos (API)")
        Ng = st.number_input("Filas", 500, 200000, 5000, 500, key="regenN")
        Sg = st.number_input("Seed", 0, 1_000_000, 42, 1, key="regenSeed")
        if st.button("🧪 Regenerar CSV de hábitos"):
            res = call_post(f"{api}/habitos/data/generar_sintetico", params={"N": int(Ng), "seed": int(Sg)}, token=api_token)
            if res.get("ok"):
                st.success("CSV regenerado.")
                st.json(res)
                st.cache_data.clear()
            else:
                st.error(res.get("error", "Error"))
