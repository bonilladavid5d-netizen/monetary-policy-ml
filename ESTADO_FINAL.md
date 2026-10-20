# ESTADO_FINAL — cierre profesional (correcciones del bloque 3 y productos finales)

- **Ejecución de cierre:** `b3_final_20261020_101239`, con `bash run_bloque2.sh b3_final_20261020_101239` en el entorno fijado del Mac (`.venv`: Python 3.13.11, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.4).
  - Código de salida: **0**. Diez etapas OK; estado `COMPLETA`.
  - Pruebas: **10/10 PASS** (1, 2, 3, 3b, 4, 5, 6, 6b, 6c, 7). Determinismo: PASS.
- **Versión:** la carpeta no es un repositorio Git; no hay commit ni push. La identificación es `MANIFEST.sha256` exterior (124 entradas) junto con el `run_id`. No se publicó nada.
- **Créditos:** David Bonilla, Yosua Ramírez y Luisa Perdomo, con la contribución declarada de David; se mantienen. La asistencia de ChatGPT y Claude Code se declara en el README.

## Qué cambió
**Código**, sin cambios de método, datos, modelos, cuadrícula ni semillas:
- `src/b2/synthetic.py`: control de selección `max(label_time usado en selección) ≤ primer estado de test`. Se deriva de los folds usados y se registra en `cortes_y_tuning.csv` (`fold = seleccion_global`, 10 filas). Los controles estrictos de ajuste no cambian. No se retiró la fila 479 ni se reestimó.
- `run_bloque2.sh`: comprueba `PY` antes de crear la corrida (si no existe, código 127 y ninguna corrida creada). Si falla el registro del entorno, la corrida queda `FALLIDA`.
- `src/b2/tests_b2.py`: nuevas pruebas 3b (selección no estricta, ajuste estricto) y 6c (intérprete inexistente en una copia temporal).
- `src/b2/figures_b2.py`: la figura 1 explica T0/T1/T2, describe los puntos como repeticiones simuladas y define el regret. El pie de la figura 2 va en tres líneas, con márgenes mayores.

**Documentación:**
- `PROTOCOLO.md`: selección, ajuste y decisión documentados por separado; magnitud de la acción (±0,50 pp en canales, ±0,05 pp en la tasa esperada siguiente).
- `AFIRMACIONES_CANDIDATAS.md` (S2): la interpretación de Ridge se sustituye por la redacción del veredicto.
- `CAMBIOS_B2.md`: anexo del bloque 3.
- `estado/MAPEO_RUTAS_MANIFEST.md`: documenta el traslado de `config_b2_usada.json` a `estado/`. El manifiesto de ejecución se conserva intacto.

**Productos nuevos:**
- `README.md` (el borrador se conserva como `README_BORRADOR.md`);
- `docs/MEMO_EJECUTIVO.md` y `.pdf`;
- `docs/NOTA_METODOLOGICA.md`;
- `docs/CV_Y_ENTREVISTA.md` (guion de 282 palabras).

## ¿Se conservaron las cifras?
Sí. Al comparar `b3_final_20261020_101239` con la entrega B2 (`b2_final_20261020_000133`), la diferencia numérica es **0** y las columnas de texto son idénticas en:
- `sintetico_por_semilla.csv`
- `sintetico_predicciones.csv`
- `real_metricas.csv`
- `real_predicciones.csv`
- `diagnostico_ipw_legacy.csv`

Cifras principales:

| Componente | Cifras |
|---|---|
| Regret, simulación | RF 0,119 · mantener 0,351 · Ridge 1,551 |
| Pérdida esperada, simulación | RF 7,284 · mantener 7,516 · óptimo 7,165 |
| Reducciones de RF frente a mantener | 66,0 % del regret y 3,1 % de la pérdida total |
| RMSE real, lag1 | Ridge 2,214 · media 2,333 · persistencia 2,948 · RF 3,317 |
| Mejoras de Ridge | 5,1 % frente a la media y 24,9 % frente a la persistencia |
| Empates de RF | concentran el 85,5 % de su regret (diagnóstico de esas filas) |

## Qué se ejecutó y qué se inspeccionó
**Ejecutado:**
- Ejecución integrada final, en un directorio nuevo (código 0).
- Comprobación manual del caso `PY=/nope/python` en una copia temporal: código 127, sin carpeta de corrida. La prueba 6c lo automatiza.
- Renderizado del memo con Chrome headless (`tools/render_memo.py`, entorno separado `.venv_docs` con markdown y pypdf): 2 páginas. *Versión sustituida después; ver «Modificaciones documentales posteriores a la ejecución».*
- `tools/verify_package.py` sobre el ZIP extraído en un directorio temporal (`outputs/audit/verificacion_paquete_final.json`):
  - manifiesto exterior 124/124, sin archivos no listados;
  - manifiesto de ejecución 61/61, con el mapeo de `config_b2_usada.json`;
  - 0 enlaces markdown internos rotos;
  - 0 archivos prohibidos (entornos, cachés, `.pyc`, ZIP, binarios de modelos) y 0 patrones de credenciales;
  - las 14 cifras principales recalculadas desde las tablas aparecen en README, memo y nota.

**Inspeccionado visualmente:**
- Figuras 1, 2 y 3 de la ejecución de cierre, sin recortes. El pie de la figura 2 es legible y completo.
- Las dos páginas del PDF del memo: tablas completas, letra de unos 9,6 pt y sin cortes.
- Detalle menor: en el pie de la figura 3 el guion bajo de `real_metricas.csv` se ve muy tenue. Esa figura no se modificó en este bloque.

- Receta desde el ZIP final extraído en una carpeta temporal: `PY=<.venv del Mac>/bin/python bash run_bloque2.sh verif_final`. Resultado: código 0, `COMPLETA` y 10/10 PASS, con diferencia 0 frente a las tablas empaquetadas (síntesis y real, métricas y predicciones). Esa corrida temporal no forma parte del paquete.

**No ejecutado:**
- B1, la búsqueda de versiones y la revisión de asesores. El ZIP se generó por última vez tras actualizar este archivo, y la verificación final se repitió sobre ese ZIP.

## Pendientes registrados al cerrar la ejecución (histórico)
1. Revisión de David del paquete final, y decisión y autorización de publicación en su repositorio. *Resuelto: publicado por autorización de David (ver abajo).*
2. Antes de publicar, decidir si se mantiene `logs/entorno.txt`, que incluye el nombre de host del Mac (`uname -a`). *Se mantiene sin editar como evidencia del entorno; está cubierto por el manifiesto de ejecución.*
3. Añadir la URL del repositorio a `docs/CV_Y_ENTREVISTA.md` cuando exista. *Resuelto.*
4. Opcional: `git init` y commit local del hito. *Resuelto con la publicación.*

## Modificaciones documentales posteriores a la ejecución (2026-10-20)
Ninguna toca el código, los datos, las tablas, las figuras, los logs ni `estado/MANIFEST_ejecucion.sha256`. Los modelos no se reestimaron.

1. **Ajustes de la revisión de cierre** (`CIERRE_MONETARIA_AJUSTES_FINALES.zip`): se integraron tres reemplazos tras comprobar sus huellas previas.
   - `docs/MEMO_EJECUTIVO.md`;
   - `docs/MEMO_EJECUTIVO.pdf`, regenerado con ReportLab en 2 páginas; sustituye al de Chrome descrito arriba, y el script `tools/render_memo.py` se conserva;
   - `docs/CV_Y_ENTREVISTA.md`.

   Corrigen tres puntos: el 94,4 % corresponde al numerador IPW de Random Forest; la falta de soporte de la acción histórica no significa que el modelo conozca el resultado; y la lectura de la acción simulada (±0,50 pp en canales, ±0,05 pp en la tasa siguiente).
2. **Frases residuales sobre la pérdida total** en `docs/NOTA_METODOLOGICA.md` §1.6 y en `docs/CV_Y_ENTREVISTA.md` (pregunta 2). De la pérdida óptima media de 7,165, los choques aportan 0,13, la penalización por mover la tasa 0,305 y la distancia esperada a las metas unos 6,73. Se recalculó desde `tablas/sintetico_predicciones.csv`.
3. **Publicación:** se añadió la URL real (https://github.com/bonilladavid5d-netizen/monetary-policy-ml) a `docs/CV_Y_ENTREVISTA.md`, se actualizó este archivo y se agregó `.gitignore`. El `MANIFEST.sha256` exterior se actualizó solo en las entradas de los archivos modificados.

