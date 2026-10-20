# Política monetaria y machine learning — revisión y protocolo del bloque 2

David Bonilla · 20 de octubre de 2026  
Entrada revisada: `HANDOFF_01.zip`, SHA-256 `a0f11f8be6c9c98881f17224e61e25f209842c969fe5133bcc406b694c7d0a9f`.

## Decisión

**El bloque 1 aporta evidencia suficiente para avanzar. No valida todavía las conclusiones económicas originales.** Se conserva la reproducción y se ejecuta una versión corregida con dos componentes:

1. **Laboratorio de decisiones:** comparar Ridge y Random Forest bajo una simulación explícita, tomando decisiones antes de conocer los choques futuros.
2. **Aplicación histórica:** comparar pronósticos macroeconómicos con referencias simples y validación temporal. La comparación de políticas reales por IPW queda como diagnóstico del método anterior, no como resultado de superioridad frente a la Reserva Federal.

Pregunta del caso: **¿Qué aportan modelos lineales y no lineales a la predicción macroeconómica y a la elección de acciones en un entorno simulado, y qué límites impiden trasladar esas decisiones a los datos históricos?**

La señal profesional buscada es definir una decisión, manejar datos y tiempo correctamente, comparar alternativas y comunicar qué evidencia permite actuar. Los resultados nuevos decidirán si algún modelo aporta valor. No se fija un ganador.

## Qué comprobé en esta revisión

Revisé código, tablas, logs y documentación. Ejecuté comprobaciones dirigidas en Python 3.12.14, NumPy 2.3.5, pandas 2.2.3 y scikit-learn 1.8.0. **No repetí aquí las quince etapas ni todos los experimentos de versiones del Mac.**

| Evidencia | Comprobación independiente | Implicación |
|---|---|---|
| Manifiesto | 36 de 36 archivos coinciden con sus huellas. | La instantánea es internamente consistente. |
| Pipeline guardado | Merge, features y tabla de escenarios coinciden byte a byte con los derivados incluidos. | Esos datos tienen una reproducción documentada. |
| Simulación v0 | Regeneré 600 filas; tratamientos idénticos y diferencia numérica máxima de 3,55 × 10⁻¹⁵ respecto de la base guardada. | Se recuperó la procedencia de la base sintética principal. |
| RF real guardado | Las tres predicciones de acción difieren como máximo 2,66 × 10⁻¹⁵, en los 35 meses. | El `argmin` distingue redondeos, no diferencias sustantivas de pérdida. |
| Reajuste dirigido de RF | Con parámetros guardados y un hilo: rango contrafactual 0 en test. Un árbol usa T1; ninguno usa T2. RMSE 3,13373 en este entorno. | No afirmar que el bosque jamás usa tratamientos. La conclusión comprobada es que no distingue las acciones en ese test. Esta ejecución no reproduce exactamente el RMSE del entorno del Mac. |
| IPW legado | Recalculé Ridge 1,13190 y RF 759,99563. Dos meses representan 94,36 % del numerador de RF. | No interpretar la diferencia como una política superior. |
| Acción expansiva real | Cero T2 observados en los 35 meses de test; “siempre T2” devuelve numéricamente 0. | El valor es no evaluable con esa comparación, no una pérdida perfecta. |
| Validación interna | Los cinco folds usan etiquetas de entrenamiento que incluyen meses posteriores al primer origen de validación. | Separar temporalmente las etiquetas de doce meses antes de ajustar hiperparámetros. |
| Simulador original | Primero genera choques y pérdidas potenciales, después usa esas pérdidas para asignar T. A estado fijo, cambiar un choque futuro cambia las probabilidades de acción. | La asignación incorpora información futura y el “óptimo” original es ex post. Corregir ambos en una versión nueva. |
| Fallo del runner | Extraje la función `step`, ejecuté un comando que sale en 7 y comprobé que continúa y sale en 0. | Un código final 0 no garantiza éxito de todas las etapas. Reparar la propagación de errores. |

El handoff también vincula la simulación v1 con la tabla de la presentación y documenta que “actualizada” y “heterogéneo” no tienen procedencia recuperada. **Se archivan esas dos variantes sin seguir buscando combinaciones de parámetros.** No bloquean el caso nuevo.

Precisiones sobre IPW:
- El estimador Horvitz–Thompson no es inválido por definición ni siempre favorece políticas con pocas coincidencias. Aquí fallan las condiciones para interpretar la comparación y el soporte es insuficiente.
- El tratamiento histórico se construye a partir del cambio mensual de la tasa efectiva promedio; no equivale a observar directamente una decisión de una reunión del FOMC.
- Ese mismo cambio está en X. Condicionado a ese X, T queda determinado; una probabilidad suavizada o recortada de un clasificador no crea soporte causal.
- Renombrar cualquier estado `evaluable_con_soporte_limitado` del diagnóstico como `NO_VALIDADO_COMO_VALOR_CAUSAL`. Si no hay coincidencias, mostrar `NA` en resultados interpretables y conservar el cero solo como evidencia del cálculo legado.
- Normalizar pesos, cambiar el umbral o resolver empates no subsana por sí solo la identificación causal.

## Alcance operativo

Claude implementa y ejecuta en la carpeta canónica existente del Mac. ChatGPT revisará `HANDOFF_02.zip`. No se crea otra copia principal ni se exige cambiar de aplicación.

Conservar `originals/`, `legacy/` y los outputs del bloque 1. Implementar módulos nuevos en `src/b2/`, configuración en `config/b2.json` y resultados en una carpeta nueva identificada por ejecución. Mantener los autores y créditos originales. David declara participación en teoría, planificación, código y análisis; eso no elimina los créditos de Yosua Ramírez y Luisa Perdomo.

No ampliar a redes neuronales, nuevos países, nuevos generadores, un dashboard ni búsqueda masiva de hiperparámetros. No descargar vintages nuevos ni actualizar las series en este bloque. Tampoco es necesario rehacer las quince etapas del bloque 1.

## A. Simulación corregida: `S0_B2_exante`

### A1. Pregunta y alcance

¿Qué modelo elige acciones con menor pérdida **esperada a un paso**, sobre estados de evaluación generados por una política de comportamiento conocida?

No es una evaluación de trayectorias contrafactuales completas ni del bienestar de largo plazo. El estado de cada mes avanza bajo la acción del simulador, no bajo cada política candidata. Reportar esta limitación.

### A2. Conservar estructura y fijar parámetros

Usar las ecuaciones y parámetros estructurales de `src/simular_datos_v0_dr2.py` como base; no cambiar signos, persistencias ni varianzas para favorecer un modelo. Escenario nuevo, nunca sobrescribir S0 o v1.

- Estado X = inflación, desempleo, tasa y actividad.
- Acciones: T0 = mantener, T1 = +0,50 puntos porcentuales, T2 = −0,50 puntos porcentuales.
- Metas del laboratorio: inflación 2; desempleo 4,5. Son supuestos de simulación, no una afirmación de metas oficiales.
- λ = 1; μ = 2; ruido = 1; 600 observaciones después de 24 de calentamiento.
- Diez semillas prefijadas: 42–51. No descartar semillas por resultados adversos.
- El criterio penaliza la **decisión** Δr², como v0; no confundirlo con la variación realizada y ruidosa de la tasa de v1.

### A3. Corregir la secuencia y el oráculo

Para cada X y acción a, obtener la media condicional de la transición ejecutando las ecuaciones con choque cero. Sean mπ y mu las medias de inflación y desempleo del siguiente periodo.

`q(X,a) = (mπ−2)^2 + σπ² + λ[(mu−4,5)^2 + σu²] + μ(Δr_a)^2`.

Incluir el factor de ruido al cuadrado en las varianzas si se expone ese parámetro. Esta esperanza es analítica porque los choques de inflación y desempleo son aditivos, de media cero y varianza conocida.

Orden obligatorio:
1. Observar X y calcular q para definir la política de comportamiento del simulador.
2. Asignar acción usando `p(a|X) = 0,9 × softmax(−1,2 × q(X,a)) + 0,1/3`. Implementar softmax estable restando el mínimo de q. Este piso de exploración es una elección nueva de diseño, no un hallazgo empírico.
3. Después generar los choques futuros. Usar flujos aleatorios independientes para inicialización, acciones y choques, derivados de la semilla.
4. Calcular resultados potenciales con choques comunes entre acciones; observar para entrenamiento solo la pérdida de la acción asignada.
5. Avanzar el estado con esa acción.

El oráculo para evaluar decisiones es `argmin_a q(X,a)`, no el mínimo de pérdidas realizadas después del choque. Guardar q y pérdidas potenciales en una tabla de verdad separada del conjunto factual. **Los modelos y la selección de hiperparámetros no pueden acceder a esa tabla.** La política generadora sí usa q por construcción; documentarlo.

### A4. Modelos y evaluación

- Ridge con X, dos indicadores de acción e interacciones X × acción; estandarización aprendida dentro de cada fold.
- RF con X y dos indicadores de acción.
- Referencias fijas: siempre T0, siempre T1 y siempre T2; oráculo ex ante como límite de referencia.
- Primer 80 % para desarrollo y último 20 % para test. Retirar una observación inmediatamente anterior al test del ajuste; hacer lo mismo en cinco folds internos expansivos (`gap=1`). Verificar por índices de tiempo que ninguna etiqueta de ajuste alcance el primer momento evaluado.
- Elegir hiperparámetros por RMSE factual de los folds, no por el regret verdadero reservado. Ajustar al desarrollo permitido y evaluar una vez por semilla. Reportar que el tuning predictivo puede no seleccionar la mejor regla de decisión.
- Usar la cuadrícula común definida abajo. RF con `random_state=42` y `n_jobs=1`; semillas 42–51 corresponden al DGP.
- Métrica decisional principal: media de `q(X,acción_elegida) − min_a q(X,a)` en test.
- También: pérdida esperada media, coincidencia con el conjunto óptimo, proporción de cada acción y empates del modelo; RMSE factual para Ridge y RF.
- Resumir los diez resultados con media, desviación y rango, más diferencias emparejadas Ridge–RF por semilla. No tratarlos como intervalos de incertidumbre sobre la economía real.
- Empates: tolerancia fija `1e-8 + 1e-6 × max(1, abs(mínimo))`. Registrar el conjunto empatado. Elegir T0 si pertenece; de lo contrario, la primera acción por orden T0/T1/T2. El desempate operativo no convierte una predicción plana en una recomendación informativa.
- Reportar la frecuencia de las acciones en cada conjunto de entrenamiento y test. No ocultar ausencia de acciones por semilla.

## B. Datos reales: evaluación predictiva retrospectiva

### B1. Pregunta y variable objetivo

¿Ridge o RF anticipan un índice futuro de inflación y desempleo mejor que dos referencias simples, con separación temporal explícita?

Usar los cuatro CSV originales de FRED: CPIAUCSL, UNRATE, FEDFUNDS e INDPRO. Reconstruir derivados desde esos insumos en cada ejecución. **El modelo debe consumir los derivados recién generados**, no una copia anterior de `data/derived/`.

Conservar el objetivo original, pero llamarlo **índice**, no pérdida de bienestar:

`Y_t = Σ(h=1…12) infl_m[t+h] + media(h=1…12) unrate[t+h]`.

`infl_m` es `100 × (CPI_t/CPI_{t−1}−1)`. La suma de tasas mensuales es una aproximación aditiva, no inflación compuesta exacta. λ = 1 es una ponderación elegida; no atribuirle preferencias estimadas de un banco central. No optimizar acciones reales contra este índice.

### B2. Información disponible

Origen de predicción: final del mes t. Convención principal: solo variables fechadas hasta t−1. Es una aproximación conservadora de disponibilidad mensual, **no una reconstrucción de calendarios de publicación ni de vintages**. Los datos revisados impiden llamarlo backtest en tiempo real.

Fijar 12 predictores:
- Cinco series: `infl_m`, `infl_yoy`, `unrate`, `fedfunds`, `indpro_g_yoy`, cada una en t−1 y t−12: diez columnas.
- Desviación estándar móvil de doce meses de `infl_m` y `unrate`, terminando en t−1, `ddof=1`: dos columnas.

No incluir tratamiento, dummies de acción, variables del mes t ni indicadores NBER determinados ex post. No usar pérdidas futuras como predictores. Las etiquetas futuras pertenecen únicamente a entrenamiento cuando ya estarían observadas y a evaluación.

### B3. Cortes, ajuste y métricas

- Usar orígenes desde enero de 1992, siempre que las doce features y Y estén completas; registrar pérdidas de filas y fechas reales sin forzar conteos.
- Mantener evaluación final de marzo de 2021 a enero de 2024: 35 orígenes si la reconstrucción coincide. **Este periodo ya fue examinado en el trabajo anterior**; describirlo como evaluación histórica reutilizada, no test virgen.
- Ajuste final fijo, sin reentrenamientos durante ese periodo. Para el primer origen t, una etiqueta de origen r solo entra si `r+12 ≤ t−1`. Por ello, el último origen permitido del entrenamiento principal es febrero de 2020.
- Usar todos los orígenes completos y elegibles hasta ese límite, incluidos los doce meses que el split legado retiraba entre train y val. Documentar que cambian features, selección temporal y universo; no atribuir todo el cambio de métricas a una sola corrección.
- Tuning con cinco folds expansivos dentro del desarrollo, sin mezclar meses. En una serie mensual completa, `TimeSeriesSplit(n_splits=5, gap=12)` implementa la separación básica. Además, exigir por fechas `máximo(label_end_train) ≤ primer_origen_validación−1 mes`.
- Imputación, escalado y cualquier aprendizaje se ajustan solo al entrenamiento del fold. Preferir no imputar; fallar ante faltantes inesperados en las cuatro series y documentar faltantes de borde por rezagos.
- Referencias: media del Y del entrenamiento permitido, fija dentro de cada bloque evaluado; y persistencia del índice retrospectivo `Σ infl_m[t−12:t−1] + media unrate[t−12:t−1]`.
- Comparar las dos referencias con Ridge y RF sobre exactamente las mismas filas: RMSE, MAE, n, fechas y mejora relativa de RMSE respecto de persistencia. R² es opcional.
- Guardar todas las predicciones de test y de los folds de desarrollo. Sus métricas sirven como diagnóstico y selección; no presentarlas como una segunda evaluación independiente después de seleccionar hiperparámetros en esos mismos folds.
- Los objetivos de meses consecutivos se solapan. No calcular intervalos o pruebas que supongan 35 observaciones independientes. No afirmar estabilidad universal a partir de este periodo.

### B4. Una sensibilidad acotada

Repetir con disponibilidad hasta t−2: desplazar **todo el conjunto informativo** un mes adicional (rezagos 2 y 13, ventanas hasta t−2, persistencia de t−13 a t−2). Exigir `label_end_train ≤ primer_origen_eval−2 meses`: gap mínimo 13 en la serie mensual y ajuste final hasta enero de 2020. Mantener las mismas fechas de test y la misma cuadrícula; seleccionar hiperparámetros solo en desarrollo. Publicar ambas variantes sin escoger la más favorable como principal.

## C. Cuadrícula y reproducibilidad comunes

Cuadrícula compacta, fijada antes de resultados B2:
- Ridge: α ∈ {0,01; 0,1; 1; 10; 100}.
- RF: 300 árboles y tres pares `(max_depth, min_samples_leaf)`: `(5,5)`, `(10,5)`, `(None,20)`.
- Selección: menor RMSE factual agregado sobre las predicciones de los folds. Empate numérico: mayor α para Ridge; en RF, menor profundidad y luego mayor tamaño de hoja. Documentar el orden.
- Congelar entorno local y registrar versiones. No repetir la búsqueda de cinco versiones de sklearn.
- Verificar determinismo repitiendo el ajuste elegido de una semilla sintética y del modelo real principal; no repetir todas las cuadrículas. Comparar predicciones, decisiones y métricas con tolerancias declaradas.

`run_bloque2.sh` debe comprobar insumos, detenerse ante el primer error, devolver código distinto de cero y no presentar salidas antiguas como nuevas. Cada ejecución usa un directorio nuevo y un identificador. Marcar completa la ejecución solo después de todos los controles. No modificar el runner de reproducción histórica sin conservar su versión.

Pruebas dirigidas obligatorias, sin inflar una suite que solo repita la implementación:
1. A X fijo y con la misma aleatoriedad de acción, cambiar choques futuros no cambia probabilidades ni acción del nuevo generador.
2. Verificar q analítico frente a pérdidas esperadas simuladas en varios estados y acciones, con tolerancia Monte Carlo justificada por su error estándar.
3. Verificar separación por fechas en todos los folds y en test, para lag 1 y lag 2.
4. Cambiar etiquetas de test o la tabla de verdad reservada no altera ajuste, hiperparámetros ni decisiones ya producidas; solo evaluación.
5. Empates cercanos generan el mismo desempate y quedan identificados. La predicción plana no desaparece del reporte.
6. Inyectar una falla en una copia temporal: código no cero, ninguna etapa posterior y ninguna entrega final marcada completa. No tocar fuentes originales para la prueba.
7. Integridad: huellas de insumos sin cambios, métricas calculables desde las predicciones entregadas, mismas filas entre modelos.

No continuar con una conclusión si falla un control material. Reparar y repetir solo la parte afectada y una ejecución final integrada.

## D. Entregables de `HANDOFF_02.zip`

Un paquete compacto y autocontenido para revisar; sin entornos virtuales, cachés, modelos binarios ni copias de los tres ZIP originales. Incluir los cuatro CSV macro originales, pequeños, con procedencia y huellas, el código B2, configuración, dependencias, receta y evidencia de la ejecución nueva. No empaquetar todas las carpetas de resultados de B1.

| Archivo / grupo | Contenido mínimo |
|---|---|
| `ESTADO.md` | Versión y huella/commit, ejecución válida, decisiones implementadas, resultados principales, fallos o limitaciones y comando exacto. Distinguir ejecutado, inspeccionado y pendiente. |
| `docs/PROTOCOLO.md` | Este diseño aplicado, parámetros congelados, fechas e información disponible; cualquier desviación y motivo. |
| `docs/CAMBIOS_B2.md` | Hallazgo → cambio → evidencia; diferencias respecto del trabajo original sin acusaciones ni atribuciones inventadas. |
| `config/b2.json`, código, `run_bloque2.sh`, dependencias | Todo lo necesario para ejecutar B2 desde los cuatro CSV y el simulador, con rutas relativas. |
| `tablas/sintetico_por_semilla.csv` | Escenario, semilla, modelo/política, n_train/test, hiperparámetros, RMSE factual cuando aplique, regret esperado, pérdida esperada, coincidencia óptima, empates y frecuencias de acción. |
| `tablas/sintetico_predicciones.csv` | Semilla, periodo, X, T observado, pérdida factual, tres predicciones, acción elegida, empates y q de evaluación; distinguir columnas que nunca entran al ajuste. |
| `tablas/real_metricas.csv` | Variante de disponibilidad, modelo, bloque, n, fechas, RMSE, MAE y mejora contra persistencia. |
| `tablas/real_predicciones.csv` | Variante, bloque/fold, fecha, Y observado, cuatro predicciones y fechas límite de información/etiquetas usadas en el ajuste. |
| `tablas/cortes_y_tuning.csv` | Rama, semilla/variante, fold, rangos train/evaluación, último mes de etiquetas train, n, hiperparámetros y métricas de selección. |
| `tablas/diagnostico_ipw_legacy.csv` | Resultado numérico legado, coincidencias, ESS, concentración de pesos y estado no causal; cero sin coincidencias marcado no evaluable. No volver a rankear políticas. |
| `tablas/pruebas_b2.csv` y `logs/` | Prueba, esperado, observado, PASS/FAIL y ruta de evidencia; log de ejecución y del fallo controlado. |
| `figuras/` | Tres figuras: regret por semilla/modelo y referencias; observado versus pronosticado en datos reales; comparación de errores y sensibilidad lag 1/2. Legibles, ejes/unidades/fuentes y sin títulos que anticipen ganador. Abrir para comprobar su renderizado. |
| `docs/AFIRMACIONES_CANDIDATAS.md` | Cada afirmación con cifra, tabla/filas, alcance y límite. Ningún resultado real como causal. |
| `README_BORRADOR.md` | Pregunta, motivación, dos ramas, hallazgos efectivos, utilidad, límites, contribución y créditos. Breve, todavía sujeto al bloque 3. |
| `MANIFEST.sha256` | Huellas del código, configuración, datos y outputs del paquete; identificador compartido de ejecución. |

El esquema de tablas puede implementarse con columnas adicionales útiles; no omitir evidencia esencial por cumplir un número arbitrario de archivos. El paquete debe permitir recalcular las métricas principales sin volver a entrenar.

## E. Condición de cierre del bloque 2

Se cierra cuando la ejecución integrada termina correctamente, los controles materiales pasan y los resultados permiten redactar afirmaciones acotadas. **Que RF o Ridge pierdan contra una referencia no es un fallo de validación.** Se reporta y se explica.

Después David sube aquí `HANDOFF_02.zip`. Entonces hacemos la revisión formal de cinco asesores, revisión cruzada con etiquetas anónimas y veredicto consolidado acordados para el bloque 3. No repetirla ahora ni ejecutarla en cada corrección.

## Prompt para Claude Code

Lee este documento completo y ejecuta el bloque 2 sobre la carpeta canónica existente de `monetary-policy-ml` en mi Mac. El bloque 1 ya tiene evidencia suficiente; no repitas su reproducción completa ni busques más variantes de simulación sin procedencia.

Implementa el protocolo A–C y entrega exactamente la evidencia funcional de D, cerrando con las condiciones de E. Usa los archivos originales y la base de código existente. La revisión detectó además información futura en la asignación del simulador, un oráculo ex post, folds internos sin purga y un runner que continúa tras fallos: estos puntos deben quedar resueltos, con pruebas dirigidas.

La aplicación real será predictiva y retrospectiva; el IPW legado se conserva solo como diagnóstico. La simulación nueva evalúa decisiones ex ante a un paso. No cambies el objetivo, escenarios, semillas o test para conseguir un ganador. Si un baseline gana, repórtalo.

Trabaja de forma autónoma sobre decisiones rutinarias de implementación. Conserva originales, resultados históricos, créditos y cambios ajenos. Puedes crear código/configuración y ejecutar el análisis y las pruebas del bloque. No publiques ni hagas push. Si la carpeta no es un repositorio, basta con el manifiesto de huellas para este bloque.

Mantén mi trabajo reconocible: reutiliza lo que sea correcto, conserva nombres y comentarios útiles, explica las decisiones concretas y elimina lenguaje genérico. No inventes experiencias personales, resultados, motivaciones ni afirmaciones de autoría exclusiva. No hace falta disimular el uso de herramientas; hace falta que yo pueda entender y defender el caso.

Usa una carpeta nueva por ejecución. Si falla una etapa material, detente, registra el error y corrige la causa antes de declarar la entrega completa. No presentes outputs viejos como regenerados. Si existe una imposibilidad real que requiera cambiar el alcance metodológico, explica el bloqueo; no lo sustituyas silenciosamente por otro experimento.

Al finalizar entrega `HANDOFF_02.zip` y un resumen corto con: comando/código de salida, pruebas, hallazgos, límites, desviaciones y archivos. No necesito una narración de todos los comandos ni nuevas preguntas de confirmación para tareas ya cubiertas por este protocolo.

## Referencias metodológicas

- scikit-learn, [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html): el parámetro gap excluye observaciones al final del entrenamiento. Aquí se deriva el número necesario de la fecha de terminación de las etiquetas; usar un split temporal por sí solo no basta.
- Federal Reserve Bank of St. Louis, [FRED API: Real-Time Periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html): distinguir datos del pasado disponibles hoy de información conocida en una fecha histórica. No se recuperan vintages en B2.
- Sachdeva, Su y Joachims (2020), [Off-policy Bandits with Deficient Support](https://arxiv.org/abs/2006.09438): el soporte es una condición material de la evaluación por ponderación. Este trabajo respalda la cautela sobre soporte; no reemplaza una identificación causal específica del caso monetario.

Las elecciones de cuadrícula, piso de exploración, diez semillas y disponibilidad mensual de B2 son decisiones de diseño propuestas en esta revisión, no requisitos impuestos por esas fuentes.
