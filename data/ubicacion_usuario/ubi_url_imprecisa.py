import requests

def obtener_ubicacion_ip():
    url = "https://ipinfo.io/json"
    response = requests.get(url)
    data = response.json()
    print("IP:", data["ip"])
    print("Ciudad:", data["city"])
    print("Región:", data["region"])
    print("País:", data["country"])
    print("Coordenadas:", data["loc"])  # lat,long

obtener_ubicacion_ip()