# PROTOCOLO aplicado — Bloque 2

Diseño: `BLOQUE_2_Politica_Monetaria.md` (revisión del 20-10-2026 sobre `HANDOFF_01.zip`).
Parámetros congelados en `config/b2.json`, copiado sin cambios en cada ejecución como `config_b2_usada.json`.
Ejecución válida: `b2_final_20261020_000133` (entrega B2). Tras las correcciones del bloque 3, la ejecución de cierre es `b3_final_20261020_101239`: mismas cifras (diferencia 0 en todas las tablas de resultados) y 10/10 pruebas.

Pregunta del caso: ¿qué aportan modelos lineales y no lineales a la predicción macroeconómica y a la elección de acciones en un entorno simulado, y qué límites impiden trasladar esas decisiones a los datos históricos?

---

## A. Laboratorio de decisiones `S0_B2_exante` (`src/b2/sim_exante.py`, `src/b2/synthetic.py`)

**Qué se evalúa.** Elegir la acción con menor pérdida *esperada a un paso*, sobre los estados que genera una política de comportamiento conocida. El estado avanza con la acción del simulador, no con la de cada política candidata. Por eso no se evalúan trayectorias contrafactuales completas ni bienestar de largo plazo.

**Estructura reutilizada sin cambios.** De `src/simular_datos_v0_dr2.py` (SHA-256 `29af6663…`, comprobado en cada ejecución) se toman `DGPParams`, `_initial_state`, `_next_state` (incluida la no linealidad `tanh` del canal tasa → inflación) y `_loss`:
- Acciones: T0 = mantener; T1 = +0,50 pp; T2 = −0,50 pp. La acción desplaza en ±0,50 pp la tasa que usan los canales de inflación, desempleo y actividad (`r̃ = r + Δr`). La transición heredada de la tasa, `r_next = 0,9·r + 0,1·(r + Δr) + choque`, mueve el valor esperado de la tasa del siguiente estado solo ±0,05 pp. Las ecuaciones se conservan; la acción no es una réplica literal de una decisión FOMC de 50 puntos básicos.
- Metas del laboratorio: π* = 2 y u* = 4,5. Son supuestos de simulación, no metas oficiales.
- λ = 1; μ = 2; ruido = 1; 600 meses después de 24 de calentamiento; inicio de fechas en 1975-01.
- La pérdida penaliza la **decisión**, μ·(Δr_a)², como en v0.
- Semillas del DGP: 42 a 51, las diez reportadas. Ninguna se descartó.

**Pérdida esperada analítica.** Sean m_π y m_u la inflación y el desempleo del periodo siguiente obtenidos con `_next_state` y choque cero:

`q(X,a) = (m_π − 2)² + (0,30·ruido)² + λ[(m_u − 4,5)² + (0,20·ruido)²] + μ(Δr_a)²`

Es exacta porque, en `_next_state`, los choques de π y u son aditivos, de media 0 y desviación σ·ruido. La prueba 2 la contrasta con Monte Carlo.

**Orden de cada periodo** (`step`):
1. Observar X_t y calcular q(X_t, ·).
2. Asignar la acción: `p(a|X) = 0,9·softmax(−1,2·q) + 0,1/3`, con softmax estable (resta del mínimo de q). La acción sale por CDF inversa de un uniforme del flujo de acciones. El piso de 0,1/3 es una elección de diseño.
3. Generar los choques del periodo, con un flujo independiente. Los flujos de inicialización, acciones y choques salen de `SeedSequence(semilla).spawn(3)`.
4. Calcular las pérdidas potenciales de las tres acciones con choques comunes. Para entrenar solo se observa la pérdida de la acción asignada.
5. Avanzar el estado con la acción asignada.

**Separación de tablas.** `sintetico/seedNN/factual.csv` (fecha, X, T, pérdida factual) es lo único que reciben `fit_and_decide()` y la selección de hiperparámetros. `verdad_reservada.csv` (q, pérdidas potenciales, p de comportamiento) solo entra en `evaluate()`. La política generadora usa q por construcción: es el «banco central» del laboratorio.

**Oráculo.** El conjunto óptimo es `{a : q_a ≤ min q + tol}`, con la misma tolerancia de empate de los modelos. El oráculo elige dentro de ese conjunto con el mismo desempate. No se usa el mínimo de pérdidas realizadas.

**Cortes y cronología** (precisada en el bloque 3; los resultados no cambian).
- La etiqueta de la fila t se conoce en t+1, porque usa X_{t+1}. Desarrollo: filas 0 a 479 (80 %). Test: filas 480 a 599 (120).
- **Selección de hiperparámetros:** `TimeSeriesSplit(5, gap=1)` sobre las filas 0 a 479. Incluye la pérdida de la fila 479, que se conoce al observar X480, el primer estado de test. La selección se cierra, por tanto, después de observar X480 y la pérdida de la transición previa, pero antes de conocer el choque que determina el resultado de la acción 480. Control no estricto, derivado de los índices usados: `max(label_time de selección) = 480 ≤ 480` (prueba 3b; fila `seleccion_global` de `cortes_y_tuning.csv`).
- **Ajuste dentro de cada fold:** separación estricta, `última_etiqueta_train ≤ primer_evaluado − 1` (prueba 3).
- **Ajuste final:** filas 0 a 478, con purga de la fila 479. La última etiqueta se conoce en 479 < 480 (control estricto).
- **Decisión en cada estado de test:** con μ̂ ya ajustado, se elige la acción observando X_t y antes de generar el choque de t.
- No es cierto que *todo* el aprendizaje termine antes de observar X480: la selección usa una etiqueta conocida justo en 480. Congelar también la selección antes de X480 sería otra convención y requeriría recalcular.

**Modelos.** La matriz de diseño es la del benchmark original (`benchmark_module.build_design_matrix`):
- Ridge: [X, T1, T2, X·T1, X·T2], con `StandardScaler` aprendido en cada ajuste.
- RF: [X, T1, T2], 300 árboles, `random_state=42`, `n_jobs=1`.
- Los hiperparámetros se eligen por RMSE factual agregado sobre las predicciones de los cinco folds. La selección minimiza el error predictivo factual y no garantiza minimizar la pérdida adicional de las decisiones; este experimento no identifica la causa del mal desempeño decisional de Ridge.
- Referencias: siempre T0, siempre T1, siempre T2 y oráculo ex ante.

**Métricas** (`tablas/sintetico_por_semilla.csv`):
- Principal: regret esperado medio, `mean(q(X, acción) − min_a q)` en test.
- También: pérdida esperada media, coincidencia con el conjunto óptimo, frecuencia de cada acción elegida, número de empates y de predicciones planas (las tres acciones empatadas), RMSE factual de test para Ridge y RF, y frecuencias de T observado en ajuste y test.
- Columnas diagnósticas añadidas: regret en las filas sin empate y regret de «siempre T0» en esas mismas filas.

**Resumen de semillas:** media, desviación, mínimo y máximo (`sintetico_resumen_semillas.csv`), y diferencias emparejadas Ridge−RF (`sintetico_diferencias_emparejadas.csv`). La columna «mejor referencia fija» se elige ex post por semilla: es un diagnóstico, no una política disponible. Nada de esto es un intervalo de incertidumbre sobre la economía real.

**Desempate** (`common.tie_choice`): tolerancia `1e-8 + 1e-6·max(1, |mínimo|)`. Se registra el conjunto empatado (por ejemplo, `T0|T1`) y se elige T0 si pertenece; si no, la primera acción en el orden T0/T1/T2. Una predicción plana sigue contándose en `n_planas` aunque el desempate le asigne una acción.

## B. Datos reales: evaluación predictiva retrospectiva (`src/b2/real.py`)

**Insumos.** Los cuatro CSV FRED originales de `data/raw/`, con huellas fijadas en la configuración. En cada ejecución se reconstruyen `base_macro_fred_unida.csv` y `base_macro_features.csv` en `<run>/derivados/` con los scripts originales `merge_bases_de_datos.py` y `03_feature_engineering.py`. El modelo consume esos derivados recién generados. En `b2_final…` y `b3_final…` coinciden byte a byte con la copia de B1 (`derivados/procedencia.json`), que no se lee.

**Objetivo (índice, no pérdida de bienestar).** `Y_t = Σ_{h=1..12} infl_m[t+h] + 1·media_{h=1..12} unrate[t+h]` (columna `loss_cum_lam100`, recalculada y verificada hasta 1e-9).
- `infl_m = 100·(CPI_t/CPI_{t−1} − 1)`. La suma de tasas mensuales aproxima de forma aditiva, no compuesta.
- λ = 1 es una ponderación elegida. No se optimizan acciones reales.

**Información disponible.** Origen al final del mes t; variante principal `lag1`, con variables fechadas hasta t−1. Es una convención conservadora, **no un calendario de publicación ni vintages**: los datos son revisados y no se puede llamar backtest en tiempo real. Los 12 predictores son:
- `infl_m`, `infl_yoy`, `unrate`, `fedfunds` e `indpro_g_yoy` en t−1 y t−12;
- la desviación estándar móvil de 12 meses (ddof = 1) de `infl_m` y `unrate`, terminando en t−1.

No entran tratamiento, dummies de acción, variables del mes t, NBER ni pérdidas futuras. Sensibilidad `lag2`: todo el conjunto informativo se desplaza un mes (rezagos 2 y 13, ventanas y persistencia hasta t−2).

**Fechas efectivas** (`estado/real_info.json`):

| | lag1 | lag2 |
|---|---|---|
| Orígenes completos desde 1992-01 | 385 | 384 (se pierde 1992-01: falta CPI de 1989-12 para `infl_yoy` en t−13) |
| Filas descartadas al final | 2024-02 → 2025-01 (Y incompleto) | igual |
| Desarrollo (ajuste final) | 1992-01 → 2020-02, n = 338 | 1992-02 → 2020-01, n = 336 |
| Última etiqueta usada | 2021-02 | 2021-01 |
| Test (evaluación histórica **reutilizada**, examinada en el trabajo anterior) | 2021-03 → 2024-01, n = 35 | igual |
| Folds | `TimeSeriesSplit(5, gap=12)`, verificación `label_end ≤ primer origen val − 1 mes` | `gap=13`, `≤ − 2 meses` |

El ajuste final es fijo: no hay reentrenamientos durante el test. Se usan todos los orígenes elegibles, incluidos los 12 que el split legado retiraba entre train y val. Respecto del trabajo original cambian a la vez las features, el corte temporal y el universo de filas. No se atribuye el cambio de métricas a una sola corrección.

**Referencias.**
- Media de Y del entrenamiento permitido, fija dentro de cada bloque: la media del fold en los folds y la del desarrollo en el test.
- Persistencia: `Σ infl_m[t−12..t−1] + media unrate[t−12..t−1]` (en lag2, de t−13 a t−2).

**Ajuste.** Ridge (escalado dentro del ajuste) y RF sobre los 12 predictores, con la cuadrícula C. No hay imputación. Si aparece un faltante en las cuatro series, el proceso falla. Los faltantes de borde se documentan arriba.

**Métricas** (`tablas/real_metricas.csv`): RMSE, MAE, n, fechas y mejora relativa de RMSE frente a persistencia, sobre las mismas filas para los cuatro pronósticos.
- `test`: evaluación histórica reutilizada.
- `folds_desarrollo_agregado` y `foldK`: **diagnóstico de selección**, no una segunda evaluación independiente.
- Los objetivos de meses consecutivos se solapan 11 meses. No se calculan intervalos ni pruebas que supongan 35 observaciones independientes.

## C. Cuadrícula y reproducibilidad

- Ridge: α ∈ {0,01; 0,1; 1; 10; 100}.
- RF: 300 árboles × (max_depth, min_samples_leaf) ∈ {(5,5), (10,5), (None,20)}.
- Selección: menor RMSE agregado de los folds. Empate (tolerancia relativa 1e-12): en Ridge gana el mayor α; en RF, la menor profundidad (None cuenta como infinita) y después la mayor hoja (`common.candidates/select`).
- Entorno congelado: Python 3.13.11, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2 y scipy 1.18.1 (`requirements.lock.txt`, `logs/entorno.txt`). No se repitió la búsqueda de versiones.
- Determinismo (`tablas/determinismo_b2.csv`): se repite dos veces el ajuste elegido de la semilla 42 y de lag1.
  - Generador: idéntico bit a bit en memoria.
  - Predicciones frente a las guardadas: diferencia máxima 1,8e-15, dentro de la tolerancia 1e-12 (redondeo del CSV).
  - Decisiones: idénticas.
- `run_bloque2.sh`:
  - comprueba insumos y huellas;
  - usa `set -euo pipefail`;
  - se niega a reutilizar un `RUN_ID`;
  - en el primer fallo escribe `estado/run_status.json = FALLIDA` con la etapa y el código, y sale con ese código;
  - marca `COMPLETA` y escribe `MANIFEST.sha256` solo en la etapa `finalize`, si las pruebas y el determinismo están en PASS.

## Desviaciones y decisiones de implementación

1. **Tolerancia del oráculo:** el conjunto óptimo ex ante usa la misma tolerancia que los modelos. El protocolo no la fijaba para q.
2. **Ridge:** el escalado se aplica a toda la matriz de diseño (incluidas dummies e interacciones), como en el benchmark original.
3. **Persistencia:** usa λ = 1 para que esté en la misma escala que Y.
4. **Tablas no exigidas pero añadidas:** `sintetico_resumen_semillas.csv`, `sintetico_diferencias_emparejadas.csv`, `determinismo_b2.csv` y `prueba2_q_montecarlo.csv`. También las columnas de regret sin empates.
5. **Runner histórico:** `run_all.sh` (B1) ahora se detiene en la primera etapa que falla. La versión anterior se conserva intacta en `run_all_B1_original.sh`.
6. **Entorno:** distinto del de la revisión (Python 3.12, sklearn 1.8.0). Las cifras de este paquete corresponden al entorno listado arriba.
7. **Ejecuciones de desarrollo:**
   - `b2_dev01` falló en `determinism`: la comparación exacta contra el CSV no admitía el redondeo de escritura. Se corrigió el criterio: igualdad bit a bit entre dos generaciones en memoria y tolerancia de 1e-12 contra el CSV.
   - `b2_dev02` falló en la prueba 6: en la copia temporal, `derive` buscaba la referencia B1 inexistente. La referencia pasó a ser opcional.
   - `b2_dev03` completó. Después se añadieron las columnas diagnósticas de empates y se ajustó la etiqueta de la figura 2.
   - La ejecución final es una corrida integrada nueva con ese código. Ninguna de las de desarrollo se presenta como resultado.

## Correcciones del bloque 3 (sin cambio de resultados)
- Cronología sintética documentada por separado (selección, ajuste y decisión) y control 3b añadido.
- Interpretación de Ridge: se elimina la atribución causal al tuning.
- Magnitud de la acción: ±0,50 pp en los canales y ±0,05 pp en la tasa esperada del siguiente estado.
- `run_bloque2.sh` comprueba el intérprete antes de crear la corrida (`PY` inexistente → código 127 y ninguna corrida creada; prueba 6c). Si falla el registro del entorno, la corrida queda FALLIDA.
- Figuras 1 y 2 rerotuladas: T0/T1/T2 explicados, puntos descritos como repeticiones y pie de la figura 2 dividido en líneas.
