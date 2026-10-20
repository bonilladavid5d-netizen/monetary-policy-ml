# Política monetaria y ML — veredicto de la revisión del bloque 3

David Bonilla · 20 de octubre de 2026  
Paquete revisado: `HANDOFF_02.zip` (104 archivos; SHA-256 `dfd783e52047e052037f2110a69b590883a09a98d05266934b03e21bf2e9d8df`).

## Veredicto del presidente

**Aprobado para pasar al cierre profesional, con correcciones acotadas. No se justifica reabrir la investigación ni añadir modelos.**

Los resultados principales tienen respaldo en los archivos: las métricas se recalculan, la simulación decide sin usar su choque futuro, el oráculo representa pérdida esperada y la rama histórica distingue predicción de recomendación causal. El proyecto todavía necesita incorporar las precisiones descritas abajo, reparar un pie de figura y completar la presentación final. Este veredicto no significa que ya esté publicado ni que se hayan revisado documentos finales que aún no existen.

La próxima intervención debe agrupar esos ajustes y los productos del bloque 4. Después basta una revisión de los cambios y del paquete final; no otra ronda completa de cinco asesores.

## Cómo se hizo la revisión

Cinco agentes elaboraron revisiones iniciales por separado: contrarian, primeros principios, expansionista, forastero y ejecutor. Después cada uno recibió las otras cuatro respuestas identificadas por letras, sin las etiquetas de autor, y escribió su contraste. El presidente revisó ambos ciclos y efectuó comprobaciones propias.

Esto cumple la revisión cruzada con autoría ocultada por etiquetas; **no supone independencia estadística entre modelos ni anonimato imposible de inferir por el contenido**. No se presenta como arbitraje humano externo.

| Asesor | Conclusión inicial | Resultado tras el contraste |
|---|---|---|
| Contrarian | Métricas correctas; precisar la frontera de selección sintética y no atribuir el mal resultado de Ridge al tuning sin un contraste. | Matiza su objeción temporal: no exige reestimar automáticamente; admite documentar la información disponible al observar X480. |
| Primeros principios | q, estimandos y cortes reales coherentes. Aclarar cronología de selección y diferencia entre desplazamiento decidido y tasa siguiente. | No identifica evidencia que invalide el resultado; distingue interpretación de implementación. |
| Expansionista | El caso gana valor si explica el intercambio económico y la magnitud incremental frente a reglas simples. | Mantener esa mejora sin añadir modelos, dashboard ni resultados causales. |
| Forastero | Falta traducir índice, regret y otros términos; figura 2 tiene pie cortado. | Prioriza comprensión, denominadores correctos y precisión temporal. |
| Ejecutor | Manifiesto exterior correcto. Dos detalles operativos y de empaquetado; no bloqueo computacional demostrado en su alcance. | Recomienda cierre focal, sin repetir B1 ni cambiar el alcance. |

Los informes iniciales y los cinco contrastes se conservan en las carpetas `asesores/` y `cruzadas/` de este paquete. Las propuestas iniciales que difieren de este veredicto quedan como historia de revisión; **la lista consolidada de este documento es la instrucción vigente**.

## Evidencia independiente

Se conservó intacta la carpeta extraída `source`; la ejecución adicional se hizo en otra copia.

- Manifiesto exterior: **103/103 huellas correctas**; el propio manifiesto es el archivo 104.
- Objetivo real recalculado directamente desde CPIAUCSL y UNRATE: diferencia máxima **3,55 × 10⁻¹⁵** frente a las etiquetas entregadas.
- RMSE/MAE reales recalculados desde las predicciones: diferencia máxima **8,88 × 10⁻¹⁶**.
- Pérdida esperada q recalculada con las ecuaciones explícitas, sin importar el evaluador del proyecto: diferencia máxima **4,26 × 10⁻¹⁴**.
- Métricas sintéticas de las 60 combinaciones semilla/política: diferencia máxima **1,78 × 10⁻¹⁵**.
- Confirmados **187 empates de RF**, 93 con las tres acciones empatadas, todos resueltos a mantener la tasa según la regla operativa.
- Abrí la figura 2 y confirmé el pie recortado. El asesor forastero abrió las tres figuras.

Además, ejecuté **la cadena B2 completa desde una copia del ZIP: código 0, diez etapas OK y ocho pruebas PASS**, incluido determinismo. En este segundo entorno se conservan las 1.200 decisiones sintéticas de cada modelo y el orden de los modelos reales; las predicciones RF presentan pequeñas diferencias entre versiones. No se exige ni se afirma igualdad binaria entre entornos.

El script y los resultados de los recálculos están en `evidencia/`. La reproducción integrada y su alcance se describen en `evidencia/REPRODUCCION_INDEPENDIENTE.md`; no confundir el entorno de esa ejecución con el entorno del Mac.

## Qué resultados podemos comunicar

Las cifras de esta sección corresponden a la ejecución entregada del Mac, `b2_final_20261020_000133`.

| Componente | Resultado | Lectura defendible |
|---|---|---|
| Laboratorio, 10 repeticiones | Regret medio: RF **0,119**, mantener **0,351**, Ridge **1,551**. | En este simulador y sus estados de evaluación, RF mejora frente a mantener y Ridge. No demuestra mejor política real. |
| Magnitud incremental | Pérdida esperada: RF **7,284**, mantener **7,516**, óptimo simulado **7,165**. | RF reduce la pérdida adicional respecto del óptimo un **66,0 %** frente a mantener, pero reduce la pérdida total un **3,1 %**. Son denominadores diferentes. |
| Pronóstico histórico principal | RMSE: Ridge **2,214**, media **2,333**, persistencia **2,948**, RF **3,317**. | Ridge tiene el menor error en las 35 fechas examinadas; mejora un **5,1 %** frente a la media y un **24,9 %** frente a persistencia. No hay prueba de superioridad general. |
| Desarrollo frente a evaluación | RF lidera los folds de desarrollo lag1, pero pierde frente a ambas referencias en la evaluación final. | Seleccionar por desarrollo no garantiza ganar en otra ventana; no se ha identificado la causa concreta de ese cambio. |

La utilidad profesional es evaluar si la complejidad de un modelo aporta una mejora relevante, establecer qué información existe antes de una decisión y saber cuándo la evidencia no permite recomendar una intervención. Esta idea debe abrir el caso; la historia de correcciones pertenece al anexo técnico.

## Decisiones finales y correcciones mínimas

### 1. Documentar la frontera temporal sintética; conservar los resultados

La fila 479 se usa en la selección de hiperparámetros; su pérdida se realiza en el estado 480, que es el primer estado de test. El ajuste final usa hasta la fila 478, cuya pérdida se conoce en 479. Las purgas de entrenamiento dentro de los folds son correctas.

**Decisión:** se permite cerrar la selección después de observar X480 y la pérdida de la transición previa, antes de conocer el choque que determina el resultado de la acción 480. Esto es coherente con la información disponible del simulador. No exige acceso a su choque futuro.

El protocolo que dimos en el bloque 2 era ambiguo: reservaba 80 % para desarrollo y retiraba una fila «del ajuste», sin precisar el último resultado usado para selección. Por eso no atribuimos este punto a una infracción inequívoca de Claude ni afirmamos que invalide el ranking.

Acciones:
- Explicar por separado selección, ajuste y decisión en `docs/PROTOCOLO.md` y en la nota metodológica final.
- Corregir cualquier frase que afirme que **todo** aprendizaje terminó antes de observar X480. La etiqueta conocida en 479 corresponde a la fila 478.
- Añadir un control pequeño que verifique `max(label_time usado para selección) <= primer_estado_test`, mientras los controles de ajuste conservan su desigualdad estricta. No probarlo solo sobre un número escrito a mano: derivarlo de los índices efectivamente usados.
- No retirar la fila 479, cambiar folds o reestimar por este motivo. Si en el futuro se exige congelar todo antes de X480, eso sería una nueva convención y requeriría recalcular.

### 2. Precisar interpretación económica y causal

- Sustituir en S2 la atribución del mal desempeño de Ridge al tuning por: **«La selección minimiza el error predictivo factual y no garantiza minimizar la pérdida adicional de las decisiones; este experimento no identifica la causa del mal desempeño decisional de Ridge».** No se evaluaron todos los α alternativos por regret y no hace falta hacerlo ahora.
- Explicar que la acción simulada desplaza en ±0,50 puntos porcentuales la tasa usada por los canales de inflación y desempleo. Sin embargo, la transición heredada `r_next = 0,9 r + 0,1(r + Δr) + choque` produce un cambio esperado de la tasa del siguiente estado de ±0,05 puntos porcentuales. Mantener las ecuaciones. No describirla como una réplica literal de una decisión FOMC de 50 puntos básicos.
- Mantener el carácter supuesto de metas y ponderaciones del simulador. El índice real no es una función de bienestar ni una medida oficial de pérdida del banco central.
- Conservar límites: datos revisados, evaluación histórica ya observada, 35 fechas cuyos horizontes comparten once meses, comparación sintética a un paso y ausencia de interpretación causal real.

### 3. Hacer legible la historia profesional

- Definir Y: suma de las tasas mensuales de inflación de los doce meses siguientes más el promedio del desempleo en esos meses; explicar que es un índice construido para el ejercicio. RMSE está en unidades de ese índice, no es un porcentaje de error.
- Definir regret como pérdida adicional respecto de la mejor acción del simulador; menor es mejor. Traducir semillas como repeticiones y reservar DGP/folds/vintages para la nota técnica o explicar los términos una vez.
- Presentar ambos porcentajes sintéticos (66,0 % y 3,1 %) con su denominador. Mantener que RF no supera toda regla fija en toda repetición: en la semilla 49, siempre T2 obtiene menor regret.
- Decir qué decisión permite el caso: conservar referencias simples, tratar Ridge como candidato para una futura validación y no trasladar automáticamente las reglas simuladas a la Fed.
- Opcional, una frase: los empates concentran aproximadamente 85,5 % del regret de RF. Es un diagnóstico de las mismas filas, no evidencia de que eliminarlas mejore una política operativa.

### 4. Reparar la figura y los enlaces

- Regenerar la figura 2 con el pie dividido en líneas y márgenes suficientes; abrir el archivo para verificarlo.
- Explicar T0/T1/T2 en la figura 1 y aclarar que sus puntos son repeticiones de simulación, no intervalos sobre la economía real.
- Integrar o enlazar las figuras desde el README final; comprobar rutas tras extraer el paquete.
- No cambiar resultados ni recalcular modelos por una corrección de rotulación.

### 5. Cerrar dos detalles de ejecución y empaquetado

- Un intérprete inexistente en `PY` devuelve 127 pero deja la corrida en `EN_CURSO`, porque la captura del entorno está fuera del manejo de fallos. Comprobar el intérprete antes de crear la corrida o capturar ese fallo y marcar `FALLIDA`. Probar este caso en una copia temporal; no debe dejar una ejecución aparentemente activa ni marcar `COMPLETA`.
- El manifiesto interno conserva `config_b2_usada.json` en la raíz original de la corrida, pero el ZIP mueve esa copia a `estado/`. Su hash es correcto. Conservar el manifiesto histórico y documentar ese traslado con el mapeo de rutas; verificar el manifiesto exterior sobre el paquete final. No reescribir evidencia histórica para simular que nunca hubo un traslado.
- Identificar `run_all.sh` como material histórico B1 incompleto en este ZIP, usado también por la prueba 6b. La receta autocontenida de B2 es `run_bloque2.sh`. No exigir reconstruir ni empaquetar todo B1.

## Qué falta para terminar el proyecto

La investigación central y la revisión formal están hechas. Falta integrar los ajustes anteriores y producir la presentación final:

1. README profesional con pregunta, resultado, utilidad, límites, figuras y reproducción.
2. Memo ejecutivo de máximo dos páginas, con recomendación sobre el uso de los modelos y evidencia, no una recomendación inventada de tasa para la Fed.
3. Nota metodológica suficiente para revisión técnica.
4. Descripción de CV amplia y adaptable y explicación de dos minutos.
5. Paquete limpio con código, fuentes, tablas, figuras, dependencias y evidencia actualizada.
6. Integración/publicación en el repositorio personal y comprobación de la portada, cuando David autorice esa operación.

No hace falta una app para cumplir ese objetivo. No se abre otro conjunto de escenarios, otra cuadrícula ni una evaluación con vintages en esta entrega.

## Prompt único para Claude Code — integrar correcciones y preparar entrega final

Trabaja en la misma carpeta canónica del proyecto monetario. Lee este veredicto completo y consulta los informes anexos solo como evidencia. Este documento consolida las decisiones; no ejecutes como obligaciones todas las recomendaciones iniciales que luego se matizaron.

La revisión formal del bloque 3 terminó. Prepara el cierre del proyecto aplicando las cinco correcciones mínimas y los entregables de esta sección. Trabaja de forma autónoma en cambios rutinarios. No publiques ni hagas push.

**A. Integración y método**

1. Conserva las ecuaciones, datos, modelos, cuadrícula, semillas, resultados históricos y créditos. Aplica las precisiones de cronología del veredicto: se acepta que la selección use la pérdida de la transición previa al observar X480; el ajuste final conserva su purga. No reestimes por ese motivo.
2. Añade el control de disponibilidad temporal de las etiquetas usadas para selección. Corrige la interpretación del tuning de Ridge, la descripción de las acciones de tasa y los términos del índice real. No añadas experimentos para justificar una causa que no se identificó.
3. Repara el registro del fallo temprano del intérprete y prueba ese caso. Distingue claramente receta B2 y referencia histórica B1. Documenta el traslado de config en el manifiesto de ejecución, conservando la evidencia original.
4. Regenera y abre las figuras afectadas por rotulación/recortes. Reutiliza las tablas existentes; no cambies valores para favorecer el relato.

**B. Productos profesionales**

5. Crea `README.md`, preservando `README_BORRADOR.md`. Debe entenderse en una primera lectura: pregunta → datos → metodología → hallazgos → utilidad → límites. Incluye las figuras, el índice Y explicado, las dos ramas diferenciadas, comparadores simples y la receta verificable. El historial de revisión va al anexo, no a la apertura.
6. Crea `docs/MEMO_EJECUTIVO.md` y su PDF de máximo dos páginas. Público: consultoría económica, strategy y analytics. Incluye pregunta, evidencia, magnitud, decisión sobre el uso del modelo y límites. Abre el PDF renderizado y revisa recortes, tamaño de letra y número de páginas. Si el entorno impide producir/verificar PDF, entrega la fuente completa y registra exactamente ese pendiente; no afirmes que lo verificaste.
7. Crea `docs/NOTA_METODOLOGICA.md`: ecuaciones, acciones decididas versus realizadas, cronología de entrenamiento/selección/decisión, q ex ante, pérdidas e índice, cortes, parámetros, empates, soporte y diagnóstico IPW legado, dependencias y límites. Vincula afirmaciones a tablas. Evita duplicar una auditoría extensa en el documento público.
8. Crea `docs/CV_Y_ENTREVISTA.md`: descripción general adaptable del proyecto, tres bullets cuantificados que no excedan la evidencia, un guion oral de unas 250–300 palabras y cinco preguntas técnicas con respuestas. No inventes una URL publicada. No atribuyas mejora real de política monetaria, retorno empresarial ni tareas personales que David no confirmó.
9. Mantén los créditos de David Bonilla, Yosua Ramírez y Luisa Perdomo y la contribución declarada de David. Conserva la transparencia sobre asistencia sin convertirla en el tema principal del caso. No atribuyas decisiones individuales nuevas por deducción ni fabriques anécdotas para que el texto parezca humano. Usa lenguaje concreto basado en los resultados y en el trabajo original.

**C. Verificación y entrega**

10. Ejecuta las pruebas dirigidas de los cambios. Como cierre de la versión integrada, ejecuta una vez `run_bloque2.sh` en un directorio nuevo con el entorno del Mac ya fijado. No repitas B1, búsquedas de versiones ni la revisión de asesores. Se conserva el método; si cambia una cifra principal inesperadamente, investiga y registra la causa antes de reemplazarla en los documentos.
11. Verifica coincidencia entre README, memo, nota, tablas y figuras, rutas internas y contenido del paquete tras extraerlo. Genera un manifiesto exterior nuevo. Incluye el pequeño registro de la reproducción independiente como evidencia separada, con su entorno distinto, sin afirmar igualdad binaria entre plataformas.
12. Entrega `ENTREGA_MONETARIA_FINAL.zip` y `ESTADO_FINAL.md`. El ZIP contiene los cuatro CSV FRED con procedencia, código/receta/dependencias, configuración, tablas/figuras, documentación final y pruebas/logs de cierre. Excluye entornos, cachés, archivos personales, credenciales, ZIP originales y binarios de modelos. La ausencia de Git en esta carpeta no impide revisar el paquete; no inicialices ni publiques repositorios ajenos.

El resumen final debe decir: qué cambió, qué se ejecutó, código de salida, pruebas, si las cifras principales se conservaron, productos finales y pendientes reales. No declares publicación hecha ni revisión visual que no hayas realizado. Si todo está listo, el siguiente paso será revisar el paquete final y publicarlo, no abrir otro bloque de investigación.
