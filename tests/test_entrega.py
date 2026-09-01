import json
import subprocess
import sys
import csv
from pathlib import Path

ROOT = Path(__file__).parents[1]

def test_dry_run_genera_tsv_de_siete_columnas(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "agente.py"), "--dry-run"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    tsv = ROOT / "dry_run" / "bitacora.tsv"
    try:
        with tsv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle, delimiter="\t"))
        assert rows[0] == ["N.º", "Ronda", "Prompt textual", "Entrada", "Salida cruda", "Veredicto", "Hora"]
        assert all(len(r) == 7 for r in rows)
        assert json.loads((ROOT / "esquema_v1_1.json").read_text())
    finally:
        # El modo local no contiene credenciales ni resultados de Gemini.
        for p in [ROOT / "dry_run" / "bitacora.tsv", ROOT / "dry_run" / "informe.md", ROOT / "dry_run" / "checkpoint.json", ROOT / "dry_run" / "configuracion.json"]:
            p.unlink(missing_ok=True)
        (ROOT / "dry_run").rmdir()
