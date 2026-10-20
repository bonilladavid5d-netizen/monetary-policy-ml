# Reproducción independiente del ZIP recibido

Comando desde una copia aislada de HANDOFF_02: `PY=python3 bash run_bloque2.sh independent_20261020`.

Resultado: código de salida 0; diez etapas OK; estado COMPLETA; ocho pruebas PASS y determinismo PASS. El código recibido no se corrigió para esta ejecución. La copia fuente de la revisión conserva sus huellas.

Entorno: Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, matplotlib 3.10.8 y scipy 1.17.0. Es distinto del Mac de la entrega (Python 3.13.11 y scikit-learn 1.9.1). Las dependencias del entorno de revisión ya estaban disponibles; no se instaló aquí el lock del Mac.

Se conservan las 1.200 acciones de Ridge y las 1.200 de RF del laboratorio. La diferencia máxima entre regrets por semilla/política es 1,67e-16. En la rama real se conserva el orden de los cuatro modelos en ambas variantes. RF cambia ligeramente: RMSE test lag1 3,316176 frente a 3,316502 entregado; lag2 3,318106 frente a 3,309286. Esto no certifica coincidencia binaria entre versiones ni sustituye las cifras de la ejecución oficial.

Los archivos `reproduccion_comparacion.json`, `pruebas_reproduccion.csv`, `determinismo_reproduccion.csv`, `etapas_reproduccion.log`, `entorno_reproduccion.txt` y las tablas de métricas adjuntas documentan esa ejecución.

El cálculo independiente de etiquetas/métricas está en `comprobacion_independiente.py`. Para repetirlo contra una extracción del HANDOFF_02 original: `python3 comprobacion_independiente.py --source /ruta/HANDOFF_02 --output /ruta/resultados_revision`. No requiere importar el evaluador del proyecto. Depende de pandas y NumPy. Esta comprobación no reentrena modelos.
