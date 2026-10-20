import csv
import os

# [PATCH-B1] carpeta absoluta original -> argv[1] (entrada FRED) y argv[2] (salida)
import sys
carpeta = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
carpeta_salida = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "repro", "pipeline")

archivos = {
    "UNRATE.csv": "unrate",
    "CPIAUCSL.csv": "cpi",
    "FEDFUNDS.csv": "fedfunds",
    "INDPRO.csv": "indpro",
}

columnas = ["date", "unrate", "cpi", "fedfunds", "indpro"]
datos = {}

for archivo, nombre_variable in archivos.items():
    ruta = os.path.join(carpeta, archivo)
    with open(ruta, "r", encoding="utf-8-sig") as f:
        lector = csv.reader(f)
        next(lector)
        for fila in lector:
            if len(fila) < 2:
                continue
            fecha = fila[0].strip()
            if not fecha:
                continue
            datos.setdefault(fecha, {"date": fecha})
            datos[fecha][nombre_variable] = fila[1].strip()

os.makedirs(carpeta_salida, exist_ok=True)
salida = os.path.join(carpeta_salida, "base_macro_fred_unida.csv")

with open(salida, "w", newline="", encoding="utf-8") as f:
    escritor = csv.DictWriter(f, fieldnames=columnas, restval="", extrasaction="ignore")
    escritor.writeheader()
    for fecha in sorted(datos):
        escritor.writerow(datos[fecha])

print("Merge completado.")
print("Archivo guardado en:", salida)
