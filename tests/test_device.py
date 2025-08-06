import os
import types

import pytest

from solhunter_zero import device as device_module


def test_detect_gpu_and_get_default_device_mps(monkeypatch):
    monkeypatch.setattr(device_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(device_module.platform, "machine", lambda: "arm64")
    torch_stub = types.SimpleNamespace(
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True, is_built=lambda: True)
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
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True, is_built=lambda: True)
        ),
        cuda=types.SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setattr(device_module, "torch", torch_stub, raising=False)
    with caplog.at_level("WARNING"):
        assert device_module.detect_gpu() is False
    assert "Rosetta" in caplog.text


def test_detect_gpu_mps_install_hint(monkeypatch, caplog):
    monkeypatch.setattr(device_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(device_module.platform, "machine", lambda: "arm64")
    torch_stub = types.SimpleNamespace(
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_built=lambda: False, is_available=lambda: False)
        ),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(device_module, "torch", torch_stub, raising=False)
    with caplog.at_level("WARNING"):
        assert device_module.detect_gpu() is False
    assert (
        "pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url https://download.pytorch.org/whl/metal"
        in caplog.text
    )


@pytest.mark.parametrize(
    "machine,is_available,expected_gpu,expected_dev,env_value",
    [
        ("arm64", True, True, "mps", "1"),
        ("arm64", False, False, "cpu", "1"),
        ("x86_64", False, False, "cpu", None),
    ],
)
def test_mps_env_and_fallback(monkeypatch, machine, is_available, expected_gpu, expected_dev, env_value):
    monkeypatch.setattr(device_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(device_module.platform, "machine", lambda: machine)
    torch_stub = types.SimpleNamespace(
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: is_available, is_built=lambda: True)
        ),
        cuda=types.SimpleNamespace(is_available=lambda: False),
        device=lambda name: types.SimpleNamespace(type=name),
    )
    monkeypatch.setattr(device_module, "torch", torch_stub, raising=False)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    assert device_module.detect_gpu() is expected_gpu
    if env_value is None:
        assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ
    else:
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == env_value
    dev = device_module.get_default_device("auto")
    assert getattr(dev, "type", None) == expected_dev
