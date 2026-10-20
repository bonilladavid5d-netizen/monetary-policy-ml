"""Punto de entrada del bloque 2 (python -I no añade src/ al path; se añade aquí)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
stage = sys.argv[1]
if stage in ("tests", "finalize"):
    from b2 import tests_b2
    tests_b2.main(stage, Path(sys.argv[2]))
else:
    from b2 import stages
    run = Path(sys.argv[2])
    stages.STAGES[stage](run, stages.load_config(run / "config_b2_usada.json"))
