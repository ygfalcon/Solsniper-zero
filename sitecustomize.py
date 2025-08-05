try:  # pragma: no cover - optional dependency
    import sqlalchemy  # type: ignore
except Exception:  # pragma: no cover
    import types, sys, importlib.machinery

    sa = types.ModuleType("sqlalchemy")
    sa.__spec__ = importlib.machinery.ModuleSpec("sqlalchemy", None)
    sys.modules.setdefault("sqlalchemy", sa)

try:  # pragma: no cover - optional dependency
    import aiohttp  # type: ignore
except Exception:  # pragma: no cover
    aiohttp_mod = types.ModuleType("aiohttp")
    aiohttp_mod.__spec__ = importlib.machinery.ModuleSpec("aiohttp", None)
    aiohttp_mod.ClientSession = object
    aiohttp_mod.TCPConnector = object
    sys.modules.setdefault("aiohttp", aiohttp_mod)

try:  # pragma: no cover - install broader stubs when available
    from tests import stubs as _stubs  # type: ignore
    _stubs.install_stubs()
except Exception:  # pragma: no cover
    pass
