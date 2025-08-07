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
    monkeypatch.setattr(device_module, "ensure_torch_with_metal", lambda: None)
    with caplog.at_level("WARNING"):
        assert device_module.detect_gpu() is False
    expected = (
        f"pip install torch=={TORCH_METAL_VERSION} "
        f"torchvision=={TORCHVISION_METAL_VERSION} "
        + " ".join(METAL_EXTRA_INDEX)
    )
    assert expected in caplog.text


def test_detect_gpu_tensor_failure(monkeypatch, caplog):
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
        assert device_module.detect_gpu() is False
    assert "Tensor operation failed" in caplog.text


def test_ensure_torch_with_metal_failure_does_not_write_sentinel(
    monkeypatch, tmp_path, caplog
):
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
    assert not sentinel.exists()


def test_ensure_torch_with_metal_rewrites_sentinel_on_version_change(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(device_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(device_module.platform, "machine", lambda: "arm64")
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("1.0.0\n1.0.0\n")
    monkeypatch.setattr(device_module, "MPS_SENTINEL", sentinel)
    monkeypatch.setattr(device_module, "torch", None, raising=False)

    def fake_run_with_timeout(cmd, timeout):
        assert not sentinel.exists()
        return device_module.InstallStatus(True, "")

    monkeypatch.setattr(device_module, "_run_with_timeout", fake_run_with_timeout)
    fake_torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True)
        )
    )
    monkeypatch.setattr(device_module.importlib, "invalidate_caches", lambda: None)
    monkeypatch.setattr(device_module.importlib, "import_module", lambda name: fake_torch)

    device_module.ensure_torch_with_metal()
    assert sentinel.exists()
    assert (
        sentinel.read_text()
        == f"{TORCH_METAL_VERSION}\n{TORCHVISION_METAL_VERSION}\n"
    )


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
    monkeypatch.delenv("SOLHUNTER_GPU_AVAILABLE", raising=False)
    monkeypatch.delenv("SOLHUNTER_GPU_DEVICE", raising=False)
    env = device_module.ensure_gpu_env()
    assert env["TORCH_DEVICE"] == "mps"
    assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    assert env["SOLHUNTER_GPU_AVAILABLE"] == "1"
    assert env["SOLHUNTER_GPU_DEVICE"] == "mps"
    assert os.environ["TORCH_DEVICE"] == "mps"
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    assert os.environ["SOLHUNTER_GPU_AVAILABLE"] == "1"
    assert os.environ["SOLHUNTER_GPU_DEVICE"] == "mps"


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
    monkeypatch.delenv("SOLHUNTER_GPU_AVAILABLE", raising=False)
    monkeypatch.delenv("SOLHUNTER_GPU_DEVICE", raising=False)
    env = device_module.ensure_gpu_env()
    assert env["SOLHUNTER_GPU_AVAILABLE"] == "0"
    assert env["SOLHUNTER_GPU_DEVICE"] == "none"
    assert "TORCH_DEVICE" not in env
    assert "TORCH_DEVICE" not in os.environ
    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ
