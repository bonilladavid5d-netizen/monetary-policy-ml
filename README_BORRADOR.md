# Política monetaria y machine learning: qué aporta un modelo y dónde se detiene la evidencia

*Borrador del bloque 2, sujeto a la revisión del bloque 3.*

## Pregunta
¿Qué aportan modelos lineales (Ridge) y no lineales (Random Forest) a la predicción macroeconómica y a la elección de acciones en un entorno simulado, y qué límites impiden trasladar esas decisiones a los datos históricos?

## Por qué dos ramas
En datos históricos no se observa qué habría pasado con otra decisión de la Fed. Además, la acción que se construía (el cambio mensual de la tasa efectiva) formaba parte de los predictores. Por eso el caso separa dos preguntas:

1. **Laboratorio de decisiones (`S0_B2_exante`).** Un simulador con ecuaciones explícitas en el que se conoce la pérdida esperada de cada acción (mantener, +0,5 pp, −0,5 pp). Los modelos aprenden solo de la pérdida de la acción observada y deciden antes de conocer los choques del mes. Se evalúan con el regret esperado a un paso frente a reglas fijas y un oráculo ex ante. Hay 10 semillas.
2. **Pronóstico histórico (FRED, EE. UU.).** Se predice un índice de inflación y desempleo de los 12 meses siguientes con información hasta t−1 (o t−2). Se compara con la media histórica y con la persistencia en 2021-03 → 2024-01. No se recomiendan acciones reales.

## Hallazgos (ejecución `b2_final_20261020_000133`)
- **Simulación:** RF obtiene un regret medio de 0,119, frente a 1,551 de Ridge y 0,351 de «siempre mantener». Gana a Ridge y a «siempre mantener» en las 10 semillas. Ridge pierde contra «siempre mantener» en todas. El 16 % de las decisiones de RF son empates que la regla resuelve hacia mantener.
- **Datos reales:** Ridge tuvo el menor RMSE (2,21), frente a media 2,33, persistencia 2,95 y RF 3,32. RF quedó por detrás de las dos referencias simples, aunque fue el mejor en los folds de desarrollo. Con información hasta t−2 el orden se mantiene.
- **Método anterior:** el «Ridge gana por IPW» del trabajo original no es interpretable como valor de una política. Se conserva como diagnóstico (`diagnostico_ipw_legacy.csv`).

## Utilidad
El caso muestra cómo definir qué información existe antes de decidir, separar los datos de entrenamiento de la verdad reservada, construir cortes temporales con purga del horizonte y comparar contra referencias simples. También muestra cómo reportar cuándo un modelo pierde contra ellas.

## Límites
- Simulación a un paso, sin trayectorias contrafactuales; las conclusiones dependen del DGP.
- Datos revisados, no vintages; los 35 meses de evaluación se solapan y ya se habían examinado antes.
- No hay inferencia causal con datos reales.

## Reproducir
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.lock.txt
bash run_bloque2.sh            # crea outputs/b2/<RUN_ID>; código != 0 ante cualquier falla
```

## Contribución y créditos
Trabajo original: David Bonilla, Yosua Ramírez y Luisa Perdomo, trabajo final de *Tópicos en IA & Economía* (Álvaro J. Riascos Villegas), abril de 2026. David declara haber participado en la teoría, la planificación, el código y el análisis. La revisión de los bloques 1 y 2 (consolidación, protocolo corregido e implementación) se hizo con asistencia de ChatGPT (revisión y protocolo) y Claude Code (implementación y ejecución), bajo la dirección de David.
