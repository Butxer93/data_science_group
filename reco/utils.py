import math

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def polyline_stub(start, end, steps=30):
    lat1, lon1 = start; lat2, lon2 = end
    return [(lat1+(lat2-lat1)*i/steps, lon1+(lon2-lon1)*i/steps) for i in range(steps+1)]

def distance_along(poly):
    d = 0.0; acc = [0.0]
    for i in range(1, len(poly)):
        d += haversine_km(poly[i-1][0], poly[i-1][1], poly[i][0], poly[i][1])
        acc.append(d)
    return acc

def point_to_segment_distance_km(p, a, b):
    latp, lonp = p; lata, lona = a; latb, lonb = b
    x = (lonp-lona) * math.cos(math.radians((lata+latb)/2)) * 111.32
    y = (latp-lata) * 110.57
    dx = (lonb-lona) * math.cos(math.radians((lata+latb)/2)) * 111.32
    dy = (latb-lata) * 110.57
    if dx==0 and dy==0: return math.hypot(x,y)
    t = max(0.0, min(1.0, (x*dx+y*dy)/(dx*dx+dy*dy)))
    projx, projy = t*dx, t*dy
    return math.hypot(x-projx, y-projy)

def distance_to_polyline_km(pt, poly):
    best = 1e9
    for i in range(1, len(poly)):
        best = min(best, point_to_segment_distance_km(pt, poly[i-1], poly[i]))
    return best
