def baseline_score(station, context):
    liters = float(context.get("liters_needed", 40))
    price_area = float(context.get("price_area_mean", station.get("price_per_liter", 0)))
    price_s = float(station.get("price_per_liter", price_area))
    delta_price = max(0, price_area - price_s)
    saving_eur = delta_price * liters
    detour = float(station.get("detour_km", 0))
    cost_detour = detour * float(context.get("eur_per_km", 0.18))
    wait = float(station.get("wait_min", 0))
    cost_wait = float(context.get("beta_wait", 0.05)) * wait
    co2 = float(station.get("co2_stop_kg", 0))
    shadow = float(context.get("shadow_price_co2", 0))
    cost_co2 = co2 * shadow
    return (context.get("w_saving",1)*saving_eur
            - context.get("w_detour",1)*cost_detour
            - context.get("w_wait",1)*cost_wait
            - context.get("w_co2",1)*cost_co2)

def rank_candidates(candidates, context, pipe=None):
    scored = []
    if pipe is not None:
        import pandas as pd
        X = pd.DataFrame([{
            "delta_price": context["price_area_mean"]-c["price_per_liter"],
            "detour_km": c["detour_km"],
            "wait_min": c["wait_min"],
            "liters_needed": context["liters_needed"],
            "brand": c.get("brand",""),
            "fuel_type": c.get("fuel_type",""),
            "vehicle_type": context["vehicle_type"],
        } for c in candidates])
        try:
            scores = pipe.predict(X)
        except:
            scores = [baseline_score(c, context) for c in candidates]
        for c, sc in zip(candidates, scores):
            out = dict(c); out["score"] = float(sc); scored.append(out)
    else:
        for c in candidates:
            out = dict(c); out["score"] = baseline_score(c, context); scored.append(out)
    return sorted(scored, key=lambda x: x["score"], reverse=True)
