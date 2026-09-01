from types import SimpleNamespace

import pytest

from agente import seleccionar_modelo, normalizar_modelo


class ErrorSDK(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.status_code = code


class FakeModels:
    def __init__(self, names, failures=()):
        self.names = names
        self.failures = dict(failures)
        self.probed = []

    def list(self):
        return [SimpleNamespace(name=n) for n in self.names]

    def generate_content(self, *, model, contents, config):
        self.probed.append(model)
        failure = self.failures.get(model)
        if failure:
            raise ErrorSDK(failure)
        return SimpleNamespace(text=" listo. ")


class FakeClient:
    def __init__(self, names, failures=()):
        self.models = FakeModels(names, failures)


def test_normaliza_prefijo_models():
    assert normalizar_modelo("models/gemini-3.1-flash-lite") == "gemini-3.1-flash-lite"


def test_prefiere_flash_lite_35():
    client = FakeClient(["models/gemini-3.1-flash-lite", "models/gemini-3.5-flash-lite"])
    assert seleccionar_modelo(client) == "gemini-3.5-flash-lite"


def test_fallback_flash_lite_31():
    client = FakeClient(["gemini-3.1-flash-lite"])
    assert seleccionar_modelo(client) == "gemini-3.1-flash-lite"


def test_fallback_si_404():
    client = FakeClient(["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
                        [("gemini-3.5-flash-lite", 404)])
    assert seleccionar_modelo(client) == "gemini-3.1-flash-lite"


def test_modelo_preferido_del_entorno_se_prueba_primero():
    client = FakeClient(["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"])
    assert seleccionar_modelo(client, "models/gemini-3.1-flash-lite") == "gemini-3.1-flash-lite"


def test_401_detiene_seleccion():
    client = FakeClient(["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
                        [("gemini-3.5-flash-lite", 401)])
    with pytest.raises(RuntimeError, match="401"):
        seleccionar_modelo(client)
    assert client.models.probed == ["gemini-3.5-flash-lite"]
