# Semana 1 · Contrato bajo ataque

La implementación usa Python 3.11+, `google-genai`, `jsonschema` y `pytest`. La evidencia canónica de `ejecucion/canonical/` es inmutable: el informe y el ZIP se derivan de ella sin consultar Gemini.

## Ejecución

```bash
./ejecutar.sh
./ejecutar.sh --audit
./ejecutar.sh --dry-run
./ejecutar.sh --new-run
```

El modo predeterminado reutiliza la ejecución canónica, valida hashes y no solicita `API_KEY`. `--audit` solo valida. `--dry-run` genera artefactos temporales locales sin red. Solo `--new-run` permite llamar a Gemini; exige `API_KEY`, crea `ejecucion/run-<identificador>/` y fija `temperature=0`, `candidate_count=1`, `seed=20260901` y `max_output_tokens=1024`. `GEMINI_MODEL` se prueba primero, pero el modelo elegido queda bloqueado durante esa corrida.

La corrida canónica conserva `ejecucion/bitacora.tsv`, `manifest.json` y el informe. El ZIP se genera con Python, entradas ordenadas, `ZIP_STORED`, permisos y fecha fija; contiene exactamente seis archivos bajo `entrega_alvarez/`.

El agente hace como mínimo 33 llamadas reales (humo + 10 + 12 + 10), más hasta dos reparaciones semánticas por salida inválida. Los reintentos de transporte por cuota/servidor son independientes.
# espe3
