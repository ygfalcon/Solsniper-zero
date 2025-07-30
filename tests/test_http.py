import importlib
import builtins


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
