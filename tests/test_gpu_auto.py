import importlib
import sys
import types


def _make_torch(available=True):
    import importlib  # noqa: E402
    torch_mod = importlib.import_module("torch")
    torch_mod.cuda.is_available = lambda: available
    return torch_mod


def _make_cupy(available=True):
    cp_mod = types.ModuleType("cupy")
    cp_mod.__spec__ = importlib.machinery.ModuleSpec("cupy", None)
    runtime = types.SimpleNamespace(getDeviceCount=lambda: 1 if available else 0)
    cp_mod.cuda = types.SimpleNamespace(runtime=runtime)
    return cp_mod


def test_simulation_auto_enables_gpu(monkeypatch):
    monkeypatch.delenv("USE_GPU_SIM", raising=False)
    torch_mod = _make_torch(True)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "cupy", _make_cupy(False))
    import solhunter_zero.simulation as sim  # noqa: E402
    importlib.reload(sim)
    assert sim.USE_GPU_SIM is True
    assert sim._GPU_BACKEND == "torch"


def test_simulation_env_disables_gpu(monkeypatch):
    monkeypatch.setenv("USE_GPU_SIM", "0")
    torch_mod = _make_torch(True)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "cupy", _make_cupy(True))
    import solhunter_zero.simulation as sim  # noqa: E402
    importlib.reload(sim)
    assert sim.USE_GPU_SIM is False
