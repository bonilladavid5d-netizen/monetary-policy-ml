# CV y entrevista

Cifras de la ejecución `b3_final_20261020_101239`; ver `tablas/`. Repositorio público: https://github.com/bonilladavid5d-netizen/monetary-policy-ml

## Descripción general (adaptable)
**Política monetaria y machine learning: decisiones simuladas y pronóstico macroeconómico.** Proyecto académico en equipo (David Bonilla, Yosua Ramírez y Luisa Perdomo; *Tópicos en IA & Economía*, 2026), revisado y reejecutado en 2026-10. Compara Ridge y Random Forest en dos tareas:
- elegir acciones de tasa en un simulador macroeconómico con pérdida esperada conocida;
- pronosticar un índice de inflación y desempleo de EE. UU. (FRED) frente a referencias simples.

Incluye cortes temporales con purga del horizonte, separación entre los datos del modelo y la verdad del evaluador, diagnóstico de un estimador IPW no interpretable y un pipeline reproducible en Python con 10 pruebas automáticas.

*Énfasis según la vacante:*
- **Consultoría económica:** el contraste entre reducir 66 % la pérdida evitable y reducir 3 % la pérdida total.
- **Strategy:** cuándo la evidencia no permite recomendar una intervención.
- **Analytics o data:** comparadores simples, validación temporal y reproducibilidad.

## Tres bullets (cuantificados y defendibles)
- Construí, en equipo, un laboratorio de decisiones de política monetaria. En 10 repeticiones, Random Forest redujo un 66 % la pérdida adicional frente a mantener la tasa (regret 0,351 → 0,119), equivalente a un 3,1 % de la pérdida total, y superó a Ridge en todas.
- Evalué pronósticos de un índice de inflación y desempleo a 12 meses con datos FRED de 1992–2024 y cortes temporales purgados. Ridge obtuvo el menor error (RMSE 2,21 frente a 2,33 del promedio histórico y 2,95 de la persistencia), y Random Forest quedó por detrás de ambas referencias.
- Auditamos y corregimos el método original: detectamos que la acción histórica quedaba determinada por los predictores y que dos observaciones concentraban el 94,4 % del numerador IPW de Random Forest. Rediseñamos la evaluación y la dejamos en un pipeline reproducible con 10 pruebas automáticas.

*(Nota: «construí» y «evalué» reflejan la participación declarada de David en teoría, planificación, código y análisis de un trabajo en equipo. Si la vacante lo pide, usar «construimos». La revisión de 2026-10 se hizo con asistencia de herramientas de IA.)*

## Guion oral (~280 palabras)
Este proyecto empezó como el trabajo final de un curso de IA y economía, con Yosua Ramírez y Luisa Perdomo. La pregunta era si un modelo de machine learning puede ayudar a elegir decisiones de política monetaria.

Al revisarlo encontramos un problema: la acción histórica quedaba determinada por los predictores. Para un mismo conjunto de características no había alternativas observadas que permitieran compararla. Además, dos meses concentraban el 94,4 % del numerador de la métrica IPW de Random Forest; Ridge solo tenía cinco coincidencias con la acción observada. Esa comparación no demostraba una política superior.

Por eso separamos el caso en dos preguntas. La primera, en un simulador donde conocemos la pérdida esperada de cada acción: si el modelo decide antes de ver el choque del mes, ¿elige bien? Ahí Random Forest redujo un 66 % la pérdida evitable frente a mantener la tasa, aunque eso es solo un 3 % de la pérdida total, porque la mayor parte de la pérdida persiste incluso con la mejor acción disponible a un paso. Ridge fue peor que mantener en todas las repeticiones.

La segunda, con datos reales de Estados Unidos: ¿quién pronostica mejor un índice de inflación y desempleo a doce meses? Ridge tuvo el menor error, pero solo un 5 % mejor que un promedio simple. Random Forest, que era el mejor en validación, quedó por detrás de las dos referencias simples en la evaluación.

La conclusión práctica es que siempre hay que comparar contra reglas simples, definir qué información existía al decidir y reconocer cuándo los datos no permiten recomendar una acción. Ninguno de los resultados justifica recomendar una tasa a la Fed. Todo está en un pipeline reproducible que se detiene ante cualquier fallo.

## Cinco preguntas técnicas
**1. ¿Por qué no usaron los datos históricos para evaluar decisiones?**
Porque la acción —mantener, subir o bajar según el cambio mensual de FEDFUNDS y un umbral de ±0,125 puntos porcentuales— quedaba determinada por los predictores. Condicionando en X, la probabilidad de cada acción es 0 o 1, y no hay soporte para comparar políticas. Además, en el test no hubo ningún mes de recorte, así que «siempre bajar» daba 0 por construcción. El IPW original se recalcula exactamente, pero queda marcado como no validado causalmente.

**2. ¿Qué es el regret y por qué 66 % y 3,1 % no se contradicen?**
El regret es la pérdida esperada adicional frente a la mejor acción del simulador. RF lo baja de 0,351 a 0,119, un 66 % de la pérdida *evitable*. La pérdida total, en cambio, solo pasa de 7,516 a 7,284 (3,1 %), porque la mayor parte persiste incluso con la mejor acción disponible a un paso (7,165 en promedio). Los choques aportan solo 0,13 de esa pérdida y la penalización por mover la tasa, unos 0,31; casi todo el resto es la distancia esperada a las metas, que una sola acción no corrige. Son dos denominadores distintos.

**3. ¿Cómo evitaron fugas temporales?**
En el pronóstico real, los predictores están fechados hasta t−1 y la etiqueta de un origen r (que usa meses hasta r+12) solo entra si r+12 ≤ t−1. La validación usa `TimeSeriesSplit` con un gap de 12 meses, verificado por fechas. En el simulador, la acción se elige antes de generar el choque; el ajuste final purga la última fila, y la selección usa como mucho una etiqueta conocida al observar el primer estado de test, con un control explícito.

**4. ¿Por qué Random Forest gana en la simulación y pierde en los datos reales?**
Son tareas distintas: decidir con una no linealidad conocida en el simulador frente a pronosticar en una ventana con el choque inflacionario de 2021–2022. En los datos reales, RF fue el mejor en la validación interna y el peor en la evaluación. Con 35 meses solapados no podemos atribuir la causa a sobreajuste o a un cambio de régimen; lo reportamos como límite.

**5. ¿Qué haría falta para recomendar una política real?**
Datos en tiempo real (vintages), una acción y una cronología claramente definidas, covariables previas a la decisión, soporte para comparar acciones, una estrategia de identificación causal y una ventana de evaluación no examinada antes. El problema no es incluir la acción como entrada de un modelo de resultados: es que quede determinada por las covariables con las que se pretende comparar alternativas. Este proyecto no tiene esos elementos, y por eso no hace esa recomendación.
