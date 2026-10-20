# Memo ejecutivo — Modelos de ML para decisiones y pronósticos de política monetaria

**Para:** equipos de consultoría económica, strategy y analytics · **Base:** proyecto de David Bonilla, Yosua Ramírez y Luisa Perdomo (2026), revisado y reejecutado en octubre de 2026 (ejecución `b3_final_20261020_101239`).

## Pregunta
¿Un modelo más complejo mejora las decisiones o los pronósticos de política monetaria lo suficiente para justificar su uso? ¿Qué evidencia se necesita antes de recomendar una acción?

## Respuesta corta
- **En un entorno simulado,** donde se conoce la pérdida esperada de cada acción, Random Forest elige mejor que mantener la tasa y que un modelo lineal.
- **En datos reales de EE. UU.,** el modelo lineal (Ridge) pronostica algo mejor que un promedio simple, y Random Forest queda por detrás de las referencias simples.
- **Ninguna rama justifica recomendar una tasa a la Reserva Federal.**

## Evidencia
**1. Laboratorio de decisiones (10 repeticiones simuladas, 120 meses de prueba cada una).** Acciones: mantener o desplazar ±0,50 pp la tasa usada por los canales del simulador. La tasa del periodo siguiente responde ±0,05 pp en esperanza. *Regret* = pérdida adicional frente a la mejor acción (menor es mejor).

| Política | Regret medio | Pérdida esperada |
|---|---|---|
| Random Forest | 0,119 | 7,284 |
| Siempre mantener | 0,351 | 7,516 |
| Ridge | 1,551 | 8,716 |
| Mejor acción posible | 0 | 7,165 |

- **Magnitud:** RF elimina el 66,0 % de la pérdida *adicional* de mantener, pero reduce la pérdida *total* solo un 3,1 %. La mayor parte de la pérdida persiste incluso con la mejor acción disponible a un paso.
- RF gana a mantener en las 10 repeticiones, aunque no a toda regla fija: en una repetición, «siempre bajar» fue mejor.
- El 85,5 % del regret de RF se concentra en decisiones donde el modelo no distingue entre acciones.
- Ridge pierde contra mantener en las 10. La selección minimiza el error predictivo factual y no garantiza minimizar la pérdida adicional de las decisiones; este experimento no identifica la causa.

**2. Pronóstico histórico (FRED, 2021-03 → 2024-01, 35 meses).** Se pronostica un índice de 12 meses (suma de inflación mensual más el desempleo promedio), con información hasta el mes anterior. Es un índice construido para el ejercicio, no una medida oficial.

| Pronóstico | Error (RMSE, puntos del índice) |
|---|---|
| Ridge | 2,214 |
| Promedio histórico | 2,333 |
| Persistencia (últimos 12 meses) | 2,948 |
| Random Forest | 3,317 |

- Ridge mejora un 5,1 % frente al promedio y un 24,9 % frente a la persistencia; el resultado se repite con información hasta dos meses antes.
- RF fue el mejor en la validación interna, pero el peor en la ventana de evaluación.

**3. Método anterior.** La comparación original de políticas sobre datos reales (IPW) no es interpretable como valor causal: la acción estaba determinada por un predictor, no hubo recortes de tasa en el periodo de prueba, y dos observaciones concentran el 94,4 % del numerador IPW de Random Forest. Se conserva solo como diagnóstico.

## Recomendación sobre el uso de los modelos
1. **Mantener referencias simples como estándar mínimo.** El promedio histórico quedó a 0,12 puntos del mejor modelo y por delante de Random Forest. Sin esa comparación, un modelo complejo parece mejor de lo que es.
2. **Usar Ridge solo como candidato para pronóstico,** sujeto a una validación con datos en tiempo real y en otra ventana. No es una herramienta de decisión.
3. **No trasladar reglas aprendidas en simulación a decisiones reales.** La comparación usa ecuaciones conocidas y evalúa un solo paso; no demuestra efectos en una economía real.
4. **Exigir antes de cualquier recomendación de acción:** información disponible antes de decidir, covariables que no determinen por completo la acción, soporte para comparar alternativas, una estrategia de identificación causal y referencias simples.

## Límites
- Datos revisados, no disponibles en tiempo real; la ventana de prueba ya se había examinado en el trabajo original.
- Son 35 meses con horizontes que se solapan 11 meses: no hay pruebas de significancia.
- La simulación es a un paso y depende de metas y pesos supuestos.
- Ningún resultado con datos reales es causal.

*Detalle técnico y tablas: `docs/NOTA_METODOLOGICA.md`, `tablas/`. Reproducción: `bash run_bloque2.sh`.*
