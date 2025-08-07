import os
import platform
import subprocess
import sys
import types

import pytest

from solhunter_zero import device as device_module
from solhunter_zero.device import (
    METAL_EXTRA_INDEX,
    TORCH_METAL_VERSION,
    TORCHVISION_METAL_VERSION,
)


def test_run_with_timeout_success(monkeypatch):
    def fake_run(cmd, check, timeout):  # pragma: no cover - simplified
        return None

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = device_module._run_with_timeout(["echo"], timeout=1)
    assert result.success is True
    assert result.message == ""


def test_run_with_timeout_timeout(monkeypatch):
    def fake_run(cmd, check, timeout):  # pragma: no cover - simplified
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = device_module._run_with_timeout(["echo"], timeout=1)
    assert result.success is False
    assert "timed out" in result.message


def test_detect_mps_install_retry(monkeypatch):
    monkeypatch.setattr(device_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(device_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(device_module, "_sentinel_matches", lambda: False)

    class Mps:
        def __init__(self, available: bool):
            self._available = available

        def is_available(self):
            return self._available

        def is_built(self):
            return True

    stub = types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=Mps(False)),
        cuda=types.SimpleNamespace(is_available=lambda: False),
        ones=lambda *a, **k: types.SimpleNamespace(cpu=lambda: None),
    )
    monkeypatch.setattr(device_module, "torch", stub, raising=False)

    called = {}

    def fake_install():
        called["install"] = True
        stub.backends.mps._available = True

    monkeypatch.setattr(device_module, "ensure_torch_with_metal", fake_install)

    assert device_module._detect_mps() is True
    assert called == {"install": True}


def test_detect_mps_sentinel_fallback(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(device_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(device_module.platform, "machine", lambda: "arm64")
    sentinel = tmp_path / "sentinel"
    monkeypatch.setattr(device_module, "MPS_SENTINEL", sentinel)
    monkeypatch.setattr(device_module, "_sentinel_matches", lambda: True)
    stub = types.SimpleNamespace(
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_built=lambda: True, is_available=lambda: False)
        ),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(device_module, "torch", stub, raising=False)

    def fail_install():  # should not be called
        raise AssertionError("install should not be attempted")

    monkeypatch.setattr(device_module, "ensure_torch_with_metal", fail_install)
    with caplog.at_level("WARNING"):
        assert device_module._detect_mps() is False
    assert str(sentinel) in caplog.text


def test_detect_cuda_success(monkeypatch):
    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        ones=lambda *a, **k: types.SimpleNamespace(cpu=lambda: None),
    )
    monkeypatch.setattr(device_module, "torch", torch_stub, raising=False)
    assert device_module._detect_cuda() is True


def test_detect_cuda_tensor_failure(monkeypatch, caplog):
    def failing_ones(*a, **k):
        raise RuntimeError("boom")

    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        ones=failing_ones,
    )
    monkeypatch.setattr(device_module, "torch", torch_stub, raising=False)
    with caplog.at_level("ERROR"):
        assert device_module._detect_cuda() is False
    assert "Tensor operation failed" in caplog.text


def test_detect_gpu_and_get_default_device_mps(monkeypatch):
    monkeypatch.setattr(device_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(device_module.platform, "machine", lambda: "arm64")
    torch_stub = types.SimpleNamespace(
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True, is_built=lambda: True)
        ),
        cuda=types.SimpleNamespace(is_available=lambda: True),
        device=lambda name: types.SimpleNamespace(type=name),
        ones=lambda *a, **k: types.SimpleNamespace(cpu=lambda: None),
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
        cuda=types.SimpleNamespace(is_available=lambda: False),
        ones=lambda *a, **k: types.SimpleNamespace(cpu=lambda: None),
    )
    monkeypatch.setattr(device_module, "torch", torch_stub, raising=False)
    with caplog.at_level("WARNING"):
        assert device_module.detect_gpu() is False
    assert "Rosetta" in caplog.text


def test_detect_mps_install_hint(monkeypatch, caplog):
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
        assert device_module._detect_mps(_attempt_install=False) is False
    expected = (
        f"pip install torch=={TORCH_METAL_VERSION} "
        f"torchvision=={TORCHVISION_METAL_VERSION} "
        + " ".join(METAL_EXTRA_INDEX)
    )
    assert expected in caplog.text


def test_detect_mps_tensor_failure(monkeypatch, caplog):
    monkeypatch.setattr(device_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(device_module.platform, "machine", lambda: "arm64")

    def failing_ones(*args, **kwargs):
        raise RuntimeError("boom")

    torch_stub = types.SimpleNamespace(
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True, is_built=lambda: True)
        ),
        cuda=types.SimpleNamespace(is_available=lambda: True),
        ones=failing_ones,
    )
    monkeypatch.setattr(device_module, "torch", torch_stub, raising=False)
    with caplog.at_level("ERROR"):
        assert device_module._detect_mps(_attempt_install=False) is False
    assert "Tensor operation failed" in caplog.text


def test_ensure_torch_with_metal_failure_marks_sentinel(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(device_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(device_module.platform, "machine", lambda: "arm64")
    sentinel = tmp_path / "sentinel"
    monkeypatch.setattr(device_module, "MPS_SENTINEL", sentinel)
    monkeypatch.setattr(device_module, "torch", None, raising=False)

    def fake_run_with_timeout(cmd, timeout):
        return device_module.InstallStatus(False, "boom")

    monkeypatch.setattr(device_module, "_run_with_timeout", fake_run_with_timeout)
    manual_cmd = (
        f"{sys.executable} -m pip install "
        f"torch=={TORCH_METAL_VERSION} torchvision=={TORCHVISION_METAL_VERSION} "
        + " ".join(METAL_EXTRA_INDEX)
    )
    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError):
            device_module.ensure_torch_with_metal()
    assert manual_cmd in caplog.text
    assert sentinel.exists()


@pytest.mark.skipif(platform.system() != "Darwin", reason="MPS is only available on macOS")
def test_configure_gpu_env_mps(monkeypatch):
    monkeypatch.setattr(device_module.platform, "system", lambda: "Darwin")
    torch_stub = types.SimpleNamespace(
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True)
        ),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(device_module, "torch", torch_stub, raising=False)
    monkeypatch.delenv("TORCH_DEVICE", raising=False)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    env = device_module.ensure_gpu_env()
    assert env["TORCH_DEVICE"] == "mps"
    assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    assert os.environ["TORCH_DEVICE"] == "mps"
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


@pytest.mark.skipif(platform.system() != "Darwin", reason="MPS is only available on macOS")
def test_configure_gpu_env_mps_unavailable(monkeypatch):
    monkeypatch.setattr(device_module.platform, "system", lambda: "Darwin")
    torch_stub = types.SimpleNamespace(
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        ),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(device_module, "torch", torch_stub, raising=False)
    monkeypatch.delenv("TORCH_DEVICE", raising=False)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    env = device_module.ensure_gpu_env()
    assert env == {}
    assert "TORCH_DEVICE" not in os.environ
    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ
