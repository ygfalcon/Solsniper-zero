import types

from solhunter_zero import device as device_module

import scripts.preflight as preflight


def test_detect_gpu_and_get_default_device_mps(monkeypatch):
    monkeypatch.setattr(device_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(device_module.platform, "machine", lambda: "arm64")
    torch_stub = types.SimpleNamespace(
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(
                is_available=lambda: True,
                is_built=lambda: True,
            )
        ),
        cuda=types.SimpleNamespace(is_available=lambda: True),
        device=lambda name: types.SimpleNamespace(type=name),
    )
    monkeypatch.setattr(device_module, "torch", torch_stub, raising=False)
    assert device_module.detect_gpu() is True
    dev = device_module.get_default_device("auto")
    assert getattr(dev, "type", None) == "mps"


def test_detect_gpu_rosetta(monkeypatch, caplog):
    monkeypatch.setattr(device_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(device_module.platform, "machine", lambda: "x86_64")
    torch_stub = types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
        cuda=types.SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setattr(device_module, "torch", torch_stub, raising=False)
    with caplog.at_level("WARNING"):
        assert device_module.detect_gpu() is False
    assert "Rosetta" in caplog.text


def test_check_gpu_env_var_message(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(preflight.platform, "system", lambda: "Darwin")
    monkeypatch.setattr("solhunter_zero.device.detect_gpu", lambda: True)
    ok, msg = preflight.check_gpu()
    assert ok is False
    assert msg == (
        "PYTORCH_ENABLE_MPS_FALLBACK=1 not set; "
        "export PYTORCH_ENABLE_MPS_FALLBACK=1"
    )
