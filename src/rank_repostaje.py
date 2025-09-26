import pandas as pd

def precio_con_tarjeta(precio_litro: float, nombre_estacion: str, df_beneficios: pd.DataFrame, tarjeta: str|None):
    if df_beneficios is None or tarjeta is None:
        return precio_litro
    # Filtra tarjeta
    sub = df_beneficios[df_beneficios["tarjeta"].astype(str).str.lower() == str(tarjeta).lower()]
    if sub.empty:
        return precio_litro
    precio = precio_litro
    for _, row in sub.iterrows():
        estaciones = str(row.get("estaciones_incluidas","")).split("|")
        estaciones = [e.strip().lower() for e in estaciones if e.strip()]
        if nombre_estacion.lower() in estaciones or not estaciones:
            # aplica descuento
            if pd.notna(row.get("centimos_litro")):
                try: precio -= float(row["centimos_litro"]) / 100.0
                except: pass
            if pd.notna(row.get("porcentaje")):
                try: precio *= max(0.0, 1.0 - float(row["porcentaje"]) / 100.0)
                except: pass
    return max(precio, 0.0)

def puntuacion_baseline(st, ctx, df_beneficios=None):
    litros = float(ctx.get("litros_necesarios", 40.0))
    precio_area = float(ctx.get("precio_area_medio", st.get("precio_litro", 0.0)))
    precio_bruto = float(st.get("precio_litro", precio_area))
    precio_net = precio_con_tarjeta(precio_bruto, st.get("nombre_estacion",""), df_beneficios, ctx.get("tarjeta"))

    delta_precio = max(0.0, precio_area - precio_net)
    ahorro = delta_precio * litros

    desvio = float(st.get("desvio_km", 0.0))
    coste_desvio = desvio * float(ctx.get("eur_por_km", 0.18))

    espera = float(st.get("minutos_espera", 0.0))
    coste_espera = float(ctx.get("beta_espera", 0.05)) * espera

    # CO2 (sintético): 2.64 kg/L para diésel/gasolina; 0 si EV (no aplica en combustible)
    co2_parada = 2.64 * litros
    coste_co2 = float(ctx.get("precio_sombra_co2", 0.0)) * co2_parada

    return (ctx.get("w_ahorro",1.0)*ahorro
            - ctx.get("w_desvio",1.0)*coste_desvio
            - ctx.get("w_espera",1.0)*coste_espera
            - ctx.get("w_co2",1.0)*coste_co2)

def rankear_candidatos(candidatos, contexto, df_beneficios=None, pipe=None):
    out = []
    if pipe is not None:
        import pandas as pd
        X = pd.DataFrame([{
            "delta_precio": contexto["precio_area_medio"] - precio_con_tarjeta(
                c.get("precio_litro", contexto["precio_area_medio"]),
                c.get("nombre_estacion",""),
                df_beneficios, contexto.get("tarjeta")
            ),
            "desvio_km": c.get("desvio_km", 0.0),
            "minutos_espera": c.get("minutos_espera", 0.0),
            "litros_necesarios": contexto["litros_necesarios"],
            "marca": c.get("marca",""),
            "tipo_combustible": c.get("tipo_combustible",""),
            "carretera": c.get("carretera","")
        } for c in candidatos])
        try:
            scores = pipe.predict(X)
        except Exception:
            scores = [puntuacion_baseline(c, contexto, df_beneficios) for c in candidatos]
        for c, sc in zip(candidatos, scores):
            d = dict(c); d["score"] = float(sc); d["precio_neto"] = precio_con_tarjeta(
                c.get("precio_litro", contexto["precio_area_medio"]),
                c.get("nombre_estacion",""), df_beneficios, contexto.get("tarjeta")
            )
            out.append(d)
    else:
        for c in candidatos:
            d = dict(c); d["precio_neto"] = precio_con_tarjeta(
                c.get("precio_litro", contexto["precio_area_medio"]),
                c.get("nombre_estacion",""), df_beneficios, contexto.get("tarjeta")
            )
            d["score"] = puntuacion_baseline(c, contexto, df_beneficios)
            out.append(d)
    return sorted(out, key=lambda x: x["score"], reverse=True)
