import sys
import types

import pytest

from scripts import preflight
from solhunter_zero import device as device_module


def _setup_env(monkeypatch, torch_version="2.1.0", torchvision_version="0.16.0", mps_built=True):
    class MPS:
        def is_built(self):
            return mps_built

        def is_available(self):
            return True

    torch_stub = types.SimpleNamespace(
        __version__=torch_version,
        backends=types.SimpleNamespace(mps=MPS()),
    )
    torchvision_stub = types.SimpleNamespace(__version__=torchvision_version)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torchvision", torchvision_stub)
    monkeypatch.setattr(device_module, "detect_gpu", lambda: True)
    monkeypatch.setattr(preflight.platform, "system", lambda: "Darwin")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def test_check_gpu_torch_version_mismatch(monkeypatch):
    _setup_env(monkeypatch, torch_version="2.0.0")
    ok, msg = preflight.check_gpu()
    assert not ok
    assert "PyTorch 2.1.0 required" in msg


def test_check_gpu_torchvision_version_mismatch(monkeypatch):
    _setup_env(monkeypatch, torchvision_version="0.15.0")
    ok, msg = preflight.check_gpu()
    assert not ok
    assert "torchvision 0.16.0 required" in msg


def test_check_gpu_mps_not_built(monkeypatch):
    _setup_env(monkeypatch, mps_built=False)
    ok, msg = preflight.check_gpu()
    assert not ok
    assert "MPS support" in msg


def test_check_gpu_success(monkeypatch):
    _setup_env(monkeypatch)
    ok, msg = preflight.check_gpu()
    assert ok
    assert "Metal GPU available" in msg
