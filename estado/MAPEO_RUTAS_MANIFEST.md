# Mapeo de rutas del manifiesto de ejecución

`estado/MANIFEST_ejecucion.sha256` es el manifiesto original de la corrida `b3_final_20261020_101239`, sin modificar. Sus rutas son relativas a `outputs/b2/b3_final_20261020_101239/`. En este paquete:

| Ruta en el manifiesto de ejecución | Ruta en el paquete |
|---|---|
| `config_b2_usada.json` | `estado/config_b2_usada.json` (copia trasladada; mismo hash) |
| `MANIFEST.sha256` (no incluido en sí mismo) | `estado/MANIFEST_ejecucion.sha256` |
| `tablas/…`, `figuras/…`, `logs/…`, `estado/…`, `derivados/…`, `sintetico/…` | misma ruta |

Archivos que el runner escribe después de `finalize` y que por eso no figuran en el manifiesto de ejecución: `estado/etapas_ok.txt`, `logs/etapas.log`, `logs/finalize.log`. Todo el paquete está cubierto por `MANIFEST.sha256` exterior.
