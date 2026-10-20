#!/usr/bin/env bash
# Bloque 1 — reproducción completa desde la copia de trabajo.
# Uso (desde la raíz del proyecto):  bash run_all.sh [etiqueta_corrida]
# Requiere .venv creado con:  python3 -m venv .venv && .venv/bin/pip install -r requirements.lock.txt
# No escribe en originals/, legacy/ ni data/: todas las salidas van a outputs/repro/<etiqueta>/.
set -euo pipefail
cd "$(dirname "$0")"

RUN="${1:-run_$(date +%Y%m%d_%H%M%S)}"
PY="${PY:-.venv/bin/python}"
OUT="outputs/repro/$RUN"
LOG="logs/$RUN"
mkdir -p "$OUT" "$LOG"

step () {  # step <nombre> <comando...>
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] >>> $name :: $*" | tee -a "$LOG/commands.log"
  local t0=$SECONDS
  if "$@" > "$LOG/$name.log" 2>&1; then
    echo "    OK  ($((SECONDS - t0)) s)" | tee -a "$LOG/commands.log"
  else
    echo "    FAIL rc=$? ($((SECONDS - t0)) s) — ver $LOG/$name.log" | tee -a "$LOG/commands.log"
  fi
}

{ echo "run=$RUN"; date; uname -a; "$PY" -c "import sys;print(sys.version)"; "$PY" -m pip freeze; } > "$LOG/environment.txt" 2>&1

# --- 1. Pipeline de datos reales (FRED descargado original -> features) -------
step 01_merge            "$PY" -I src/merge_bases_de_datos.py data/raw "$OUT/pipeline"
step 02_features         "$PY" -I src/03_feature_engineering.py "$OUT/pipeline/base_macro_fred_unida.csv" "$OUT/pipeline/base_macro_features.csv"

# --- 2. Simuladores (semilla 42, parámetros del __main__ de cada versión) -----
step 03_sim_v1_final     "$PY" -I src/simular_datos.py       "$OUT/sim_v1_final"
step 04_sim_v0_dr2       "$PY" -I src/simular_datos_v0_dr2.py "$OUT/sim_v0_dr2"

# --- 3. Benchmark principal ---------------------------------------------------
step 05_bench_synth_saved  "$PY" -I src/benchmark_module.py --mode synthetic --input data/derived/base_sintetica.csv       --output "$OUT/outputs_synth"
step 06_bench_synth_v0     "$PY" -I src/benchmark_module.py --mode synthetic --input "$OUT/sim_v0_dr2/base_sintetica.csv" --output "$OUT/outputs_synth_from_sim_v0_dr2"
step 07_bench_synth_v1     "$PY" -I src/benchmark_module.py --mode synthetic --input "$OUT/sim_v1_final/base_sintetica.csv" --output "$OUT/outputs_synth_from_sim_v1_final"
step 08_bench_real         "$PY" -I src/benchmark_module.py --mode real      --input data/derived/base_macro_features.csv --output "$OUT/outputs_real"

# --- 4. Robustez (módulos del final 2.zip interno) ----------------------------
step 09_regime_synth       "$PY" -I src/benchmark_regime_aware.py --mode synthetic --input data/derived/base_sintetica.csv --output "$OUT/outputs_synth_regime" --regime high_inflation
step 10_regime_real_rec    "$PY" -I src/benchmark_regime_aware.py --mode real --input data/derived/base_macro_features.csv --output "$OUT/outputs_real_regime" --regime recession
step 11_regime_real_inf    "$PY" -I src/benchmark_regime_aware.py --mode real --input data/derived/base_macro_features.csv --output "$OUT/outputs_real_regime_inflation" --regime high_inflation
step 12_lambda             "$PY" -I src/benchmark_lambda_sensitivity.py --input data/derived/base_macro_features.csv --output "$OUT/outputs_lambda"

# --- 5. Interpretabilidad / reporting -----------------------------------------
step 13_report_synth       "$PY" -I src/interpretability_reporting.py --mode synthetic --input_dir "$OUT/outputs_synth" --output_dir "$OUT/reporting_synth" --data data/derived/base_sintetica.csv
step 14_report_real        "$PY" -I src/interpretability_reporting.py --mode real      --input_dir "$OUT/outputs_real"  --output_dir "$OUT/reporting_real"  --data data/derived/base_macro_features.csv

# --- 6. Módulo treatments (escenarios por niveles de FEDFUNDS) ----------------
step 15_treatments         "$PY" -I src/aplicacion_contrafactuales.py data/derived/base_macro_features.csv "$OUT/pipeline/scenarios_tratamientos_monetarios.csv"

echo "Salidas: $OUT   Logs: $LOG"
