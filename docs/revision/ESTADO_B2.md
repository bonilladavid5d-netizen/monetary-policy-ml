# ESTADO — Bloque 2

- **Versión:** la carpeta no es un repositorio Git; no hay commit ni push. La identificación es `MANIFEST.sha256` (código, configuración, datos y outputs del paquete) junto con el `run_id`.
- **Ejecución válida:** `b2_final_20261020_000133`, en estado `COMPLETA`, con 8/8 pruebas en PASS (`tablas/pruebas_b2.csv`) y determinismo en PASS.
- **Comando exacto** (desde la raíz del proyecto): `bash run_bloque2.sh b2_final_20261020_000133`, con código de salida 0 y 10 etapas OK (`logs/etapas.log`).
- **Entorno:** Python 3.13.11, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2 y scipy 1.18.1 (`logs/entorno.txt`, `requirements.lock.txt`).
- **Créditos:** David Bonilla, Yosua Ramírez y Luisa Perdomo (trabajo original); se mantienen.

## Ejecutado
- **Etapas:** inputs (huellas de los 4 CSV FRED y del simulador v0), derive (merge y features regenerados; byte-idénticos a B1), synthetic (10 semillas), real (lag1 y lag2), cortes, legacy_ipw, determinism, figures, tests y finalize.
- **Pruebas** (`tablas/pruebas_b2.csv`):
  1. Invariancia de la acción a choques futuros.
  2. q analítico frente a Monte Carlo (max |z| = 1,67).
  3. Separación por fechas en 432 cortes.
  4. Perturbar test o verdad no cambia ajuste ni decisiones.
  5. Empates.
  6. Falla inyectada (código 1, FALLIDA, sin etapas posteriores).
  - 6b. `run_all.sh` corregido se detiene.
  7. Integridad (huellas iguales; métricas recalculadas desde las predicciones, diferencia ≤ 8,9e-16; mismas filas entre modelos).
- **Ejecuciones de desarrollo** (no se presentan como resultado):
  - `b2_dev01`: FALLIDA en determinism; criterio de comparación contra CSV corregido.
  - `b2_dev02`: FALLIDA en tests (prueba 6: referencia B1 obligatoria en la copia temporal); corregida.
  - `b2_dev03`: COMPLETA, con código anterior a las columnas diagnósticas de empates.

## Decisiones implementadas
Simulador ex ante con flujos independientes, oráculo `argmin q`, tabla de verdad reservada, gap=1 sintético, gap=12 y 13 reales con verificación por fechas, 12 predictores hasta t−L sin tratamiento, referencias de media y persistencia, cuadrícula C con desempates de hiperparámetros, desempate de acciones con registro de empates, `n_jobs=1`, IPW legado como diagnóstico y runner que se detiene ante fallos. El detalle está en `docs/PROTOCOLO.md` y `docs/CAMBIOS_B2.md`.

## Resultados principales
- **Sintético:** regret esperado medio en 10 semillas de RF 0,119, siempre T0 0,351, siempre T1 0,814, siempre T2 0,976 y Ridge 1,551. RF < Ridge y RF < siempre T0 en 10/10 semillas. Ridge > siempre T0 en 10/10.
- **Real lag1 (test 2021-03 → 2024-01, n = 35):** RMSE de Ridge 2,214, media 2,333, persistencia 2,948 y RF 3,317.
- **Real lag2:** RMSE de Ridge 2,075, media 2,336, persistencia 3,116 y RF 3,309.
- **Afirmaciones con límites:** `docs/AFIRMACIONES_CANDIDATAS.md`.

## Fallos o limitaciones
- Una referencia simple gana a un modelo: «siempre T0» supera a Ridge en el laboratorio, y la media y la persistencia superan a RF en datos reales. Se reporta como resultado, no como fallo de validación.
- RF sintético: 187 de 1200 decisiones de test son empates (93 planas) resueltos hacia T0.
- La selección de hiperparámetros es predictiva y no garantiza la mejor regla de decisión. RF eligió (10,5) en las 10 semillas; el α de Ridge varió entre 0,01 y 100.
- El test real se reutiliza (ya examinado), tiene objetivos solapados y usa datos revisados sin vintages.
- Simulación a un paso, sin trayectorias contrafactuales.
- Las cifras corresponden al entorno indicado; el de la revisión (sklearn 1.8.0) no se replicó.

## Inspeccionado, no ejecutado
- Las variantes sintéticas «actualizada» y «heterogéneo» están archivadas sin procedencia y no se buscaron.
- Las 15 etapas de B1 no se repitieron. `run_all.sh` solo se corrigió y se probó en su función `step` (prueba 6b).

## Pendiente
- Bloque 3: revisión formal de cinco asesores sobre `HANDOFF_02.zip` (no se ejecuta aquí).
- Opcional: `git init` local y commit del hito.
