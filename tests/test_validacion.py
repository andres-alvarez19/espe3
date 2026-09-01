import json
from pathlib import Path
from agente import ALLOWED_ACTIONS, validate_text

ROOT = Path(__file__).parents[1]

def test_esquema_estricto_tiene_restricciones():
    s = json.loads((ROOT / "esquema_v1_1.json").read_text())
    assert s["additionalProperties"] is False
    assert s["properties"]["severity"]["enum"] == ["baja", "media", "alta", "critica"]
    assert s["properties"]["confidence"]["minimum"] == 0
    assert s["properties"]["confidence"]["maximum"] == 1
    assert set(s["properties"]["next_action"]["enum"]) == ALLOWED_ACTIONS
    assert s["properties"]["result"]["minLength"] == 1

def test_validacion_laxa_y_estricta():
    loose = json.loads((ROOT / "esquema_v1_0.json").read_text())
    strict = json.loads((ROOT / "esquema_v1_1.json").read_text())
    raw = '{"task":"x","result":"y","severity":"urgente","confidence":2,"next_action":"hacer"}'
    assert validate_text(raw, loose)[1] is None
    assert validate_text(raw, strict)[0] is None
