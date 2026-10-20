# CAMBIOS_B2 — hallazgo → cambio → evidencia

Base: trabajo original de David Bonilla, Yosua Ramírez y Luisa Perdomo (*Tópicos en IA & Economía*, abril de 2026), reproducido en el bloque 1 (`HANDOFF_01.zip`). El código original queda intacto en `src/` y en `legacy/`; el bloque 2 añade `src/b2/`. Las diferencias descritas son de diseño metodológico. No atribuyen intención ni error a ninguna persona.

| # | Hallazgo (fuente) | Cambio en B2 | Evidencia |
|---|---|---|---|
| 1 | El simulador v0 genera primero los choques del mes y las pérdidas realizadas. Después asigna T con un softmax sobre esas pérdidas más ruido, de modo que la asignación usa información futura (revisión B2; `simular_datos_v0_dr2.py`) | Nuevo generador `S0_B2_exante`: la acción depende solo de q(X,a), y los choques se generan después, en un flujo aleatorio independiente. Ecuaciones y parámetros estructurales sin cambios | Prueba 1: con X fijo, 20 choques distintos dan una diferencia máxima de p igual a 0 y la misma acción; p recalculada desde X coincide con la guardada (2,8e-15). `src/b2/sim_exante.py:step` |
| 2 | El «óptimo verdadero» original es `argmin` de las pérdidas realizadas, es decir, ex post | Oráculo ex ante `argmin_a q(X,a)`, con q analítico. La métrica es el regret esperado | Prueba 2: q frente a Monte Carlo (N = 200 000) en 12 casos de estado y acción, con max \|z\| = 1,67 (umbral 4). `tablas/prueba2_q_montecarlo.csv` |
| 3 | Las columnas de verdad (`loss_T*`, `true_optimal_treatment`) estaban en el mismo CSV que los datos de entrenamiento. La separación dependía solo de la lista de columnas del benchmark, que no las usaba como X | Tabla factual y tabla de verdad separadas. `fit_and_decide()` no recibe la verdad; `evaluate()` es la única que la usa | Prueba 4: alterar las etiquetas de test y permutar q no cambia hiperparámetros, μ̂ ni decisiones; sí cambia el regret |
| 4 | Folds internos sin purga: `TimeSeriesSplit` sin gap, con etiquetas a 12 meses (B1 §4.3; revisión) | Real: `gap = 11 + L` y verificación por fechas `label_end ≤ primer origen val − L meses`. Sintético: `gap=1` y verificación por índices | Prueba 3: 420 cortes sintéticos y 12 reales, todos OK. `tablas/cortes_y_tuning.csv` |
| 5 | El bloque val legado (57 meses) no seleccionaba nada, y había 12 meses retirados entre train y val (B1) | Desarrollo único con todos los orígenes elegibles: 1992-01 → 2020-02 (n = 338). La selección se hace por folds de ese desarrollo | `estado/real_info.json` |
| 6 | `dfedfunds` define T y está en X; `fedfunds_t` y `fedfunds_lag1` la reconstruyen; X incluye variables del mes t y NBER ex post (B1 §4.3) | La rama real es predictiva: 12 predictores fechados hasta t−1, sin tratamiento ni NBER. Sensibilidad con t−2 | `src/b2/real.py:build_features`; la prueba 3 comprueba que las X son los valores en t−L |
| 7 | IPW real no interpretable: T está determinado por X, 0 T2 en test, pesos recortados y el RF no distingue acciones (B1 §4.2) | El IPW legado se conserva solo como diagnóstico. Estados `NO_VALIDADO_COMO_VALOR_CAUSAL` y `NO_EVALUABLE_SIN_COINCIDENCIAS`; `valor_interpretable = NA`. No se rankea | `tablas/diagnostico_ipw_legacy.csv`: Ridge 1,1319 y RF 759,9956 recalculados = guardados; 2 observaciones = 94,4 % del numerador RF; rango de μ̂_RF legado 1,8e-15 |
| 8 | `argmin` sobre valores prácticamente iguales: el RF legado decidía por redondeo con `n_jobs=-1` (B1) | Desempate explícito con tolerancia; conjunto empatado registrado; `n_jobs=1` | Prueba 5 y `n_empates`/`n_planas` en `sintetico_por_semilla.csv` (RF: 187 filas empatadas, 93 planas de 1200); determinismo: diferencia máxima 1,8e-15 y decisiones idénticas |
| 9 | El runner B1 registraba FAIL y continuaba con código final 0 (revisión) | `run_bloque2.sh` se detiene en la primera falla, devuelve el código, marca FALLIDA y solo escribe COMPLETA/MANIFEST tras las pruebas. `run_all.sh` corregido; versión previa en `run_all_B1_original.sh` | Prueba 6: con falla inyectada en una copia temporal, código 1, FALLIDA en `synthetic`, sin etapas posteriores ni manifiesto. Prueba 6b: `run_all.sh` sale con 7 y se detiene |
| 10 | Derivados leídos de una copia anterior | Merge y features se ejecutan en cada corrida desde los 4 CSV con huellas verificadas | `derivados/procedencia.json`; prueba 7 (huellas al inicio y al final iguales) |
| 11 | Pérdida real llamada «pérdida» con λ como preferencia | Se llama «índice Y»; λ = 1 es una ponderación elegida | `docs/PROTOCOLO.md` §B |
| 12 | Cuadrícula RF original 4×3 con un `n_jobs` no determinista | Cuadrícula compacta congelada antes de resultados B2 y reglas de desempate de hiperparámetros explícitas | `config/b2.json`; `cortes_y_tuning.csv` (filas `agregado`) |

Diferencias con el trabajo original que **no** son correcciones de error, sino cambios de alcance:
- La aplicación histórica ya no compara políticas (Ridge frente a RF por IPW) sino pronósticos frente a referencias simples.
- La simulación usa un nuevo escenario (`S0_B2_exante`). Los conjuntos S0 y S1 originales se conservan como resultados históricos, y las variantes «actualizada» y «heterogéneo» quedan archivadas sin procedencia.

## Anexo — correcciones del bloque 3 (`VEREDICTO_Y_PROMPT_CIERRE.md`)

| Hallazgo de la revisión | Cambio | Evidencia |
|---|---|---|
| La selección sintética usa la pérdida de la fila 479, conocida en X480; el protocolo B2 era ambiguo | Se documenta la convención aceptada; no se reestima; control `max(label_time selección) ≤ primer estado test` derivado de los folds usados | Prueba 3b; `cortes_y_tuning.csv` (`fold = seleccion_global`) |
| S2 atribuía el resultado de Ridge al tuning | Frase sustituida por la redacción del veredicto | `AFIRMACIONES_CANDIDATAS.md` S2; `PROTOCOLO.md` §A |
| Magnitud de la acción no explicada | ±0,50 pp en canales y ±0,05 pp en la tasa esperada siguiente | `PROTOCOLO.md` §A; `NOTA_METODOLOGICA.md` |
| `PY` inexistente dejaba la corrida en EN_CURSO (código 127) | Comprobación del intérprete antes de crear la corrida; registro FALLIDA si falla el entorno | Prueba 6c (`logs/prueba6_fallo_controlado/py_inexistente.txt`) |
| Pie de la figura 2 recortado; T0/T1/T2 sin explicar en la figura 1 | Figuras rerotuladas, revisadas visualmente | `figuras/` de `b3_final_20261020_101239` |
| `config_b2_usada.json` movido a `estado/` en el paquete | Se conserva el manifiesto de la corrida y se documenta el mapeo | `estado/MAPEO_RUTAS_MANIFEST.md` |
