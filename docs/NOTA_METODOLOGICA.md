# Nota metodológica

Ejecución de referencia: `b3_final_20261020_101239`. Configuración congelada: `config/b2.json` (copia de la corrida en `estado/config_b2_usada.json`). Las rutas de tablas son relativas a la raíz del paquete.

## 1. Laboratorio `S0_B2_exante`

### 1.1 Ecuaciones (sin cambios respecto de `src/simular_datos_v0_dr2.py`)
Estado X_t = (π_t, u_t, r_t, y_t): inflación, desempleo, tasa y actividad. Para la acción a con desplazamiento Δr_a ∈ {0; +0,50; −0,50} y los parámetros de `DGPParams`:

```
r̃_t   = r_t + Δr_a                                   (tasa usada por los canales)
gap_t  = r̃_t − 2,5
β_eff  = −0,35 · (1 + 0,5·tanh(π_t − 2))              (no linealidad tasa → inflación)
π_{t+1} = 0,75(π_t − 2) + 2 + β_eff·gap_t + 0,25·y_t + ε_π      ε_π ~ N(0; 0,30²)
u_{t+1} = 0,85(u_t − 4,5) + 4,5 − 0,30·y_t + 0,20·gap_t + ε_u   ε_u ~ N(0; 0,20²)
y_{t+1} = 0,70·y_t − 0,40·gap_t + ε_y                           ε_y ~ N(0; 0,25²)
r_{t+1} = 0,90·r_t + 0,10·r̃_t + ε_r                            ε_r ~ N(0; 0,15²)
```

**Acción decidida frente a tasa realizada.** La acción desplaza ±0,50 pp la tasa r̃ que entra en los canales de π, u e y. En la ecuación heredada de la tasa solo se transmite el 10 %, así que E[r_{t+1}] cambia ±0,05 pp. Se conserva la ecuación; la acción no es una réplica literal de una decisión FOMC de 50 pb.

### 1.2 Pérdida y pérdida esperada ex ante
- Pérdida realizada: `L = (π_{t+1} − 2)² + λ(u_{t+1} − 4,5)² + μ(Δr_a)²`, con λ = 1 y μ = 2. La penalización recae sobre la decisión Δr, no sobre la tasa realizada. Metas y pesos son supuestos del laboratorio.
- Pérdida esperada: `q(X,a) = (m_π − 2)² + 0,30² + λ[(m_u − 4,5)² + 0,20²] + μ(Δr_a)²`, donde m_π y m_u son π_{t+1} y u_{t+1} con choque cero. Es exacta porque los choques de π y u son aditivos y de media cero. La prueba 2 la contrasta con Monte Carlo (N = 200 000, max |z| = 1,67; `tablas/prueba2_q_montecarlo.csv`).
- **Regret** = q(X, acción elegida) − min_a q(X,a): pérdida esperada adicional frente a la mejor acción del simulador.

### 1.3 Orden de un periodo y política de comportamiento
1. Se observa X_t y se calcula q(X_t,·).
2. Se asigna la acción con `p(a|X) = 0,9·softmax(−1,2·q) + 0,1/3`.
3. Se genera el choque de t, en un flujo aleatorio independiente.
4. Se calculan las pérdidas potenciales de las tres acciones con choques comunes.
5. El estado avanza con la acción asignada.

Los modelos solo ven `sintetico/seedNN/factual.csv`. La verdad (q, pérdidas potenciales, p) está en `verdad_reservada.csv` y solo la lee el evaluador (prueba 4). Hay 24 meses de calentamiento, 600 meses guardados y semillas 42 a 51.

### 1.4 Cronología: selección, ajuste y decisión
La etiqueta de la fila t (su pérdida) se conoce en t+1.

| Paso | Filas | Última etiqueta usada se conoce en | Control |
|---|---|---|---|
| Selección de hiperparámetros (5 folds, gap = 1) | 0 a 479 | 480 = primer estado de test | `≤ 480` (no estricto), prueba 3b |
| Ajuste dentro de cada fold | según fold | primer evaluado − 1 | estricto, prueba 3 |
| Ajuste final | 0 a 478 | 479 | estricto, `< 480` |
| Decisión en X_480 … X_599 | — | — | antes del choque de cada mes |

La selección se cierra tras observar X_480 y la pérdida de la transición 479 → 480, sin acceso al choque que determina el resultado de la acción 480. Por eso **no** todo el aprendizaje termina antes de observar X_480.

### 1.5 Modelos, referencias, empates
- **Ridge:** [X, T1, T2, X·T1, X·T2], estandarizado dentro de cada ajuste.
- **RF:** [X, T1, T2], 300 árboles, `random_state=42`, `n_jobs=1`.
- **Cuadrícula:** α ∈ {0,01; 0,1; 1; 10; 100}; (profundidad, hoja) ∈ {(5,5), (10,5), (None,20)}.
- **Selección:** menor RMSE factual agregado sobre los folds. En empate gana el mayor α, la menor profundidad y la mayor hoja.
- La selección minimiza el error predictivo factual y no garantiza minimizar la pérdida adicional de las decisiones; este experimento no identifica la causa del mal desempeño decisional de Ridge.
- **Referencias:** siempre T0, T1 o T2 y el oráculo ex ante (argmin q).
- **Empates de acción:** tolerancia `1e-8 + 1e-6·max(1,|mín|)`. Si T0 está en el conjunto empatado se elige T0; si no, la primera acción en orden. El conjunto queda registrado (`sintetico_predicciones.csv`, columnas `*_empate`).
- **Resultado de los empates:** RF tiene 187 filas empatadas de 1200, 93 de ellas planas, que concentran el 85,5 % de su regret. Ridge no tiene ninguna (`sintetico_por_semilla.csv`).

### 1.6 Resultados (`tablas/sintetico_resumen_semillas.csv`, `sintetico_diferencias_emparejadas.csv`)

| Política | Regret medio (desv.; rango) | Pérdida esperada media |
|---|---|---|
| RF | 0,119 (0,098; 0,007–0,293) | 7,284 |
| Siempre T0 | 0,351 (0,115; 0,202–0,528) | 7,516 |
| Siempre T1 | 0,814 (0,545; 0,260–1,950) | 7,978 |
| Siempre T2 | 0,976 (0,586; 0,023–1,787) | 8,140 |
| Ridge | 1,551 (0,369; 0,854–2,055) | 8,716 |
| Oráculo | 0 | 7,165 |

- **Frente a mantener**, RF reduce el regret un 66,0 %: (0,351 − 0,119)/0,351, cuyo denominador es la pérdida *adicional* de mantener. La pérdida total baja un 3,1 %: (7,516 − 7,284)/7,516, cuyo denominador es la pérdida *total*. La mayor parte de esa pérdida persiste incluso con la mejor acción disponible a un paso: el óptimo ex ante promedia 7,165. De esos 7,165, las varianzas de los choques aportan solo 0,30² + 0,20² = 0,13 y la penalización por mover la tasa cuando el óptimo lo hace, 0,305 en promedio. El resto, unos 6,73, es la distancia esperada del estado a las metas, que una sola acción no elimina (recálculo desde `tablas/sintetico_predicciones.csv`, columnas `eval_q_*`).
- RF < siempre T0 y RF < Ridge en 10/10 repeticiones; Ridge > siempre T0 en 10/10. En la semilla 49, siempre T2 (0,023) < RF (0,093).

## 2. Datos históricos: pronóstico retrospectivo

### 2.1 Índice Y
`Y_t = Σ_{h=1..12} infl_m[t+h] + media_{h=1..12} unrate[t+h]`, con `infl_m = 100·(CPI_t/CPI_{t−1} − 1)`. Es la columna `loss_cum_lam100` del script original `03_feature_engineering.py`, recalculada en cada corrida y verificada hasta 1e-9.
- Es un índice construido para el ejercicio, con λ = 1 como ponderación elegida. No es una función de bienestar ni una medida oficial.
- La suma de tasas mensuales es una aproximación aditiva.
- RMSE y MAE están en puntos de Y.

### 2.2 Información y cortes (`estado/real_info.json`, `tablas/cortes_y_tuning.csv`)
- **Predictores (12):** `infl_m`, `infl_yoy`, `unrate`, `fedfunds` e `indpro_g_yoy` en t−L y t−L−11, más la desviación estándar móvil de 12 meses (ddof = 1) de `infl_m` y `unrate` hasta t−L. L = 1 es la variante principal y L = 2 la sensibilidad.
- No entran el tratamiento, variables del mes t ni NBER.
- **Etiquetas:** la del origen r se conoce en r+12 y entra al ajuste solo si `r+12 ≤ t_eval − L`.
- **Desarrollo:** 1992-01 → 2020-02 (n = 338) en lag1 y 1992-02 → 2020-01 (n = 336) en lag2.
- **Test:** 2021-03 → 2024-01 (n = 35), con un solo ajuste fijo.
- **Folds:** `TimeSeriesSplit(5, gap=11+L)` con verificación por fechas (prueba 3).
- **Referencias:** media del Y de entrenamiento (fija en cada bloque) y persistencia `Σ infl_m[t−L−11..t−L] + media unrate[t−L−11..t−L]`.

### 2.3 Resultados (`tablas/real_metricas.csv`)

| | Ridge | Media | Persistencia | RF |
|---|---|---|---|---|
| RMSE test, lag1 | 2,214 | 2,333 | 2,948 | 3,317 |
| MAE test, lag1 | 1,616 | 1,907 | 2,234 | 2,361 |
| RMSE test, lag2 | 2,075 | 2,336 | 3,116 | 3,309 |
| RMSE folds de desarrollo, lag1 (diagnóstico de selección) | 1,704 | 2,113 | 1,859 | 1,670 |

- Ridge mejora un 5,1 % frente a la media y un 24,9 % frente a la persistencia (lag1).
- Hiperparámetros elegidos: Ridge α = 100; RF (5,5) en lag1 y (10,5) en lag2.
- Las métricas de folds sirvieron para seleccionar y no son una evaluación independiente.

## 3. Diagnóstico IPW legado (`tablas/diagnostico_ipw_legacy.csv`)
Las cifras del trabajo original (Ridge 1,132; RF 759,996; «siempre T0» 815,7) se recalculan exactamente con `benchmark_module._ipw_value`. Todas quedan marcadas `NO_VALIDADO_COMO_VALOR_CAUSAL` y `valor_interpretable = NA`, por cuatro motivos:
- T, definido por el cambio de FEDFUNDS, estaba determinado por un predictor.
- El test tiene 0 meses T2: «siempre T2» = 0 por construcción y es no evaluable.
- Dos pesos recortados (1/0,001) generan el 94,4 % del numerador de RF.
- μ̂_RF tenía un rango de 1,8e-15 entre acciones.

No se rankean políticas.

## 4. Reproducibilidad
- Runner: `run_bloque2.sh`.
  - Comprueba el intérprete, los insumos y sus huellas, y se niega a reutilizar un RUN_ID.
  - Se detiene en la primera etapa que falle con su código y la marca FALLIDA.
  - Solo marca COMPLETA tras las pruebas.
- Pruebas: 10/10 PASS (`tablas/pruebas_b2.csv`). Cubren choques futuros, q frente a Monte Carlo, separación temporal (3 y 3b), aislamiento de la verdad, empates, fallo inyectado (6), intérprete inexistente (6c), runner histórico (6b) e integridad (7).
- Determinismo: dos repeticiones dan una diferencia máxima de 1,8e-15 (`tablas/determinismo_b2.csv`).
- Dependencias: `requirements.lock.txt` (Python 3.13.11, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2).
- Otro entorno (3.12 / 1.8.0, revisión independiente) conservó las decisiones del laboratorio y el orden real; RF varía ligeramente (`evidencia_revision_independiente/`).

## 5. Limitaciones
1. Datos revisados, sin vintages: no es un backtest en tiempo real; t−1 y t−2 son convenciones conservadoras.
2. La ventana 2021-03 → 2024-01 ya se había examinado en el trabajo original.
3. Son 35 orígenes con 11 meses de horizonte compartido; no se calculan intervalos ni pruebas.
4. La simulación es a un paso: el estado avanza con la acción del simulador, no con la de cada política.
5. Los resultados dependen de las ecuaciones, metas y pesos supuestos.
6. Sin inferencia causal con datos reales.
