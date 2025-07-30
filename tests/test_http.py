import importlib
import builtins
import asyncio
import pytest


def test_json_fallback(monkeypatch):
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "orjson":
            raise ModuleNotFoundError
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    import solhunter_zero.http as http
    http = importlib.reload(http)

    data = http.dumps({"a": 1})
    assert http.loads(data)["a"] == 1
    assert not http.USE_ORJSON


def test_connector_limit_env(monkeypatch):
    monkeypatch.setenv("HTTP_CONNECTOR_LIMIT", "5")
    import solhunter_zero.http as http
    http = importlib.reload(http)
    assert http.CONNECTOR_LIMIT == 5


@pytest.mark.asyncio
async def test_get_session_singleton(monkeypatch):
    import solhunter_zero.http as http
    http = importlib.reload(http)
    await http.close_session()
    s1 = await http.get_session()
    s2 = await http.get_session()
    assert s1 is s2
    await http.close_session()


def test_module_imports(monkeypatch):
    import solhunter_zero.http as http
    import solhunter_zero.jito_stream as js
    assert js.get_session is http.get_session
