#!/usr/bin/env bash
# Bloque 2 — ejecución integrada. Uso (desde la raíz):  bash run_bloque2.sh [RUN_ID]
# - comprueba insumos y huellas antes de empezar;
# - se detiene en la PRIMERA etapa que falle y devuelve su código (distinto de cero);
# - cada ejecución escribe en un directorio nuevo outputs/b2/<RUN_ID> (se niega a reutilizarlo);
# - la ejecución solo queda marcada COMPLETA tras pasar todas las pruebas (etapa finalize).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"
RUN_ID="${1:-b2_$(date +%Y%m%d_%H%M%S)}"
RUN="outputs/b2/$RUN_ID"
if [ -e "$RUN" ]; then echo "ERROR: $RUN ya existe; use un RUN_ID nuevo" >&2; exit 2; fi
# [B3] el intérprete se comprueba ANTES de crear la corrida: un PY inexistente no deja una
# ejecución aparentemente activa (EN_CURSO).
if ! "$PY" -c "import sys" > /dev/null 2>&1; then
  echo "ERROR: el intérprete PY='$PY' no existe o no se ejecuta; no se crea la corrida" >&2; exit 127
fi
for f in config/b2.json src/b2_run.py data/raw/CPIAUCSL.csv data/raw/UNRATE.csv data/raw/FEDFUNDS.csv data/raw/INDPRO.csv src/simular_datos_v0_dr2.py; do
  [ -f "$f" ] || { echo "ERROR: falta insumo $f" >&2; exit 3; }
done
mkdir -p "$RUN/logs" "$RUN/estado"
cp config/b2.json "$RUN/config_b2_usada.json"
printf '{"run_id": "%s", "estado": "EN_CURSO", "inicio": "%s"}\n' "$RUN_ID" "$(date -u +%FT%TZ)" > "$RUN/estado/run_status.json"
if ! { echo "run_id=$RUN_ID"; date; uname -a; "$PY" -c "import sys; print(sys.version)"; "$PY" -m pip freeze; } > "$RUN/logs/entorno.txt" 2>&1; then
  printf '{"run_id": "%s", "estado": "FALLIDA", "etapa_fallida": "entorno", "codigo": 1}\n' "$RUN_ID" > "$RUN/estado/run_status.json"
  echo "ERROR: no se pudo registrar el entorno (ver logs/entorno.txt)" >&2; exit 1
fi

stage () {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] >>> $name" | tee -a "$RUN/logs/etapas.log"
  local rc=0
  "$@" > "$RUN/logs/$name.log" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "    FALLO etapa=$name rc=$rc (ver logs/$name.log); ejecución detenida" | tee -a "$RUN/logs/etapas.log" >&2
    printf '{"run_id": "%s", "estado": "FALLIDA", "etapa_fallida": "%s", "codigo": %d}\n' "$RUN_ID" "$name" "$rc" > "$RUN/estado/run_status.json"
    exit "$rc"
  fi
  echo "    OK" | tee -a "$RUN/logs/etapas.log"
  echo "$name" >> "$RUN/estado/etapas_ok.txt"
}

for s in inputs derive synthetic real cortes legacy_ipw determinism figures tests finalize; do
  stage "$s" "$PY" -I src/b2_run.py "$s" "$RUN"
done
echo "Ejecución $RUN_ID COMPLETA: $RUN"
