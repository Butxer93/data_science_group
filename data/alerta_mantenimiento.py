def generar_alerta_html(kilometraje, meses):
    """
    Genera un div HTML con avisos de mantenimiento según kilometraje o tiempo.
    kilometraje: int -> kilómetros recorridos
    meses: int -> tiempo desde último mantenimiento en meses
    """

    alertas = []

    # Grupo corto plazo (10k - 20k km o 6 meses)
    if kilometraje % 10000 < 1000 or meses >= 6:
        alertas.append("Cambio de aceite, filtro de aceite, rotación de neumáticos")

    # Grupo medio plazo (30k - 60k km o 2 años)
    if kilometraje % 30000 < 1000 or meses >= 24:
        alertas.append("Filtro de combustible, líquido de frenos, líquido refrigerante, filtro de aire")

    # Grupo largo plazo (80k - 120k km o 5 años)
    if kilometraje % 80000 < 2000 or meses >= 60:
        alertas.append("Correa de distribución, amortiguadores, discos de freno")

    # Construcción del div en HTML
    if alertas:
        html = f"""
        <div style='padding:15px; background-color:#ffcccc; border:1px solid #cc0000; border-radius:8px;'>
            <h3>⚠️ Aviso de mantenimiento</h3>
            <ul>
                {''.join(f"<li>{a}</li>" for a in alertas)}
            </ul>
        </div>
        """
    else:
        html = """
        <div style='padding:15px; background-color:#ccffcc; border:1px solid #009900; border-radius:8px;'>
            <h3>✅ Todo en orden</h3>
            <p>No corresponde mantenimiento por ahora.</p>
        </div>
        """

    return html


# Ejemplo de uso
print(generar_alerta_html(kilometraje=30500, meses=25))
