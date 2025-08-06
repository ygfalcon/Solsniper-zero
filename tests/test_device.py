import types

from solhunter_zero import device as device_module


def test_detect_gpu_and_select_device_mps(monkeypatch):
    monkeypatch.setattr(device_module.platform, "system", lambda: "Darwin")
    torch_stub = types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
        cuda=types.SimpleNamespace(is_available=lambda: False),
        device=lambda name: types.SimpleNamespace(type=name),
    )
    monkeypatch.setattr(device_module, "torch", torch_stub, raising=False)
    assert device_module.detect_gpu() is True
    dev = device_module.select_device("auto")
    assert getattr(dev, "type", None) == "mps"
