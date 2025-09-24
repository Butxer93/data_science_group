from plyer import gps

def obtener_ubicacion(lat, lon):
    print(f"Latitud: {lat}, Longitud: {lon}")

# Inicializar GPS
gps.configure(on_location=obtener_ubicacion)
gps.start(minTime=1000, minDistance=1) # milisegundos y metros

# Requiere los permisos del XML "permisos_ubi"