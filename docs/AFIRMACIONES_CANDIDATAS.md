# Afirmaciones candidatas — Bloque 2

Ejecución: `b2_final_20261020_000133`; cifras idénticas en la ejecución de cierre `b3_final_20261020_101239`. Cada afirmación indica su cifra, la tabla y las filas que la respaldan, su alcance y su límite. Ninguna afirmación sobre datos reales es causal. Todas siguen sujetas a la revisión del bloque 3.

## Laboratorio sintético `S0_B2_exante` (decisión ex ante a un paso)

**S1. En este laboratorio, Random Forest elige acciones con menor regret esperado que Ridge en las 10 semillas.**
- Cifra: regret medio RF 0,119 (desviación 0,098; rango 0,007–0,293) frente a Ridge 1,551 (0,369; 0,854–2,055). Diferencia emparejada Ridge−RF entre 0,756 y 1,982.
- Tablas: `sintetico_resumen_semillas.csv` (filas `rf`, `ridge`); `sintetico_diferencias_emparejadas.csv`.
- Alcance: este DGP (ecuaciones v0, no linealidad tanh), 120 meses de test por semilla y estados generados por la política de comportamiento.
- Límite: no evalúa trayectorias ni bienestar de largo plazo. Las 10 semillas describen variabilidad del simulador, no incertidumbre sobre la economía.

**S2. Ridge, con interacciones lineales X×acción, rinde peor que la regla fija «siempre T0» en las 10 semillas.**
- Cifra: Ridge 1,551 frente a siempre T0 0,351. Exceso de Ridge entre 0,652 y 1,528 por semilla.
- Ridge recomienda T0 el 6,6 % de las veces, frente al 38,9 % del oráculo, y coincide con el conjunto óptimo el 4,5 % de las veces.
- Tablas: `sintetico_por_semilla.csv` (`politica ∈ {ridge, siempre_T0, oraculo_exante}`).
- Límite: la selección minimiza el error predictivo factual y no garantiza minimizar la pérdida adicional de las decisiones; este experimento no identifica la causa del mal desempeño decisional de Ridge. El α elegido varía entre semillas (0,01 a 100).

**S3. RF supera a «siempre T0» en las 10 semillas, pero no a la mejor regla fija en todas.**
- Cifra: ventaja frente a siempre T0 entre 0,045 y 0,466 por semilla. En la semilla 49, «siempre T2» (0,023) tiene menor regret que RF (0,093).
- Tablas: `sintetico_diferencias_emparejadas.csv`, columna `mejor_referencia_fija`.
- Límite: la mejor regla fija se identifica ex post por semilla. No es una política disponible antes de ver el test.

**S4. Una parte de las decisiones de RF son desempates, no recomendaciones informativas.**
- Cifra: 187 de 1200 filas de test con empate en μ̂ (93 con las tres acciones iguales). Todas se resuelven hacia T0 por la regla operativa.
- En las 1013 filas sin empate, el regret medio de RF es 0,020, frente a 0,295 de siempre T0 en esas mismas filas (RF menor en las 10 semillas).
- Tablas: `sintetico_por_semilla.csv` (columnas `n_empates`, `n_planas`, `regret_esperado_medio_filas_sin_empate`, `regret_siempre_T0_mismas_filas_sin_empate`); `sintetico_predicciones.csv` (`rf_empate`).
- Límite: el regret total de RF incluye el beneficio de que el desempate elija T0, una acción que el oráculo elige el 38,9 % de las veces.

**S5. La pérdida esperada media con RF (7,284) queda a 0,119 del límite del oráculo (7,164).**
- Tabla: `sintetico_resumen_semillas.csv` (`perdida_esperada_media_mean`).
- Límite: son unidades de pérdida del simulador. No se comparan con el índice Y real.

## Aplicación histórica (pronóstico retrospectivo, sin acciones)

**R1. Con información hasta t−1, Ridge tuvo el menor error de pronóstico del índice Y en 2021-03 → 2024-01.**
- Cifra: RMSE 2,214 (MAE 1,616), frente a la media de entrenamiento 2,333 (1,907), persistencia 2,948 (2,234) y RF 3,317 (2,361). La mejora de Ridge es del 24,9 % frente a persistencia y del 5,1 % frente a la media.
- Tabla: `real_metricas.csv` (`variante=lag1`, `bloque=test`); `real_predicciones.csv`.
- Alcance: 35 orígenes con objetivos solapados 11 meses y un solo ajuste fijo.
- Límite: el periodo ya se había examinado en el trabajo anterior (evaluación reutilizada, no test virgen). Los datos son revisados, sin vintages. No hay prueba de significancia ni afirmación de estabilidad.

**R2. Random Forest pronosticó peor que las dos referencias simples en el periodo de evaluación, con ambas convenciones de información.**
- Cifra: RMSE de RF 3,317 (lag1) y 3,309 (lag2), frente a persistencia 2,948 y 3,116 y media 2,333 y 2,336.
- Tabla: `real_metricas.csv` (`bloque=test`).
- Límite: en los folds de desarrollo, RF lag1 tuvo el menor RMSE (1,670, frente a Ridge 1,704 y persistencia 1,859). El orden entre modelos cambia entre desarrollo y evaluación; con 35 meses solapados no se puede separar si eso se debe a un cambio de régimen (inflación 2021–2022) o a sobreajuste. Las métricas de folds son diagnóstico de selección.

**R3. El resultado de R1 se mantiene al desplazar la información a t−2.**
- Cifra: Ridge 2,075, media 2,336, persistencia 3,116 y RF 3,309. Ridge mejora un 33,4 % frente a persistencia.
- Tabla: `real_metricas.csv` (`variante=lag2`, `bloque=test`); `figuras/fig3_real_errores_lag1_lag2.png`.
- Límite: es una sola sensibilidad. Ambas variantes se publican y ninguna se elige como principal por su resultado.

**R4. Ninguna referencia simple domina en todos los periodos.**
- Cifra: la media de entrenamiento supera a la persistencia en test (RMSE 2,333 frente a 2,948), pero no en los folds de desarrollo (2,113 frente a 1,859).
- Tabla: `real_metricas.csv`.
- Límite: es descriptivo.

## Diagnóstico del método anterior (no es un resultado de superioridad)

**L1. Las cifras IPW legadas (Ridge 1,132; RF 759,996) se recalculan exactamente, pero no se validan como valor causal.**
- Detalle: 0 meses T2 en test, por lo que «siempre T2» no es evaluable. Dos observaciones explican el 94,4 % del numerador IPW de RF. El rango de μ̂_RF entre acciones en el test legado es 1,8e-15. T está determinado por `dfedfunds`, que forma parte de X.
- Tabla: `diagnostico_ipw_legacy.csv`.
- Límite: no se rankean políticas. Normalizar pesos o cambiar umbrales no resolvería la identificación.

## Lo que el paquete NO permite afirmar
- Que algún modelo recomiende mejores decisiones a la Reserva Federal.
- Que Ridge pronostique mejor en general o en tiempo real.
- Que el ranking sintético se traslade a la economía de EE. UU.
