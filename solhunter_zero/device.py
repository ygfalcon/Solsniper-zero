from __future__ import annotations

import argparse
import importlib
import logging
import os
import platform
import subprocess
import sys

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch is optional at runtime
    torch = None  # type: ignore


def configure_gpu_env(_attempt_install: bool = True) -> bool:
    """Detect GPU support and configure environment variables.

    The function prefers Apple's Metal backend (MPS) on macOS machines with
    Apple Silicon and falls back to CUDA on other platforms.  When a GPU
    backend is detected, ``TORCH_DEVICE`` is set accordingly and, for MPS,
    ``PYTORCH_ENABLE_MPS_FALLBACK`` is set to ``"1"``.  If no GPU is found a
    warning is logged and ``TORCH_DEVICE`` is set to ``"cpu"``.  The return
    value indicates whether a GPU backend was configured.
    """

    env: dict[str, str] = {}
    logger = logging.getLogger(__name__)

    if torch is None:
        logger.warning("PyTorch is not installed; falling back to CPU")
        env["TORCH_DEVICE"] = "cpu"
        os.environ.update(env)
        return False

    def _install_and_retry(reason: str) -> bool:
        logger.warning("%s; attempting to install MPS-enabled PyTorch", reason)
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch==2.1.0",
            "torchvision==0.16.0",
            "--extra-index-url",
            "https://download.pytorch.org/whl/metal",
        ]
        try:
            subprocess.check_call(cmd)
            logger.info("PyTorch installation succeeded")
            importlib.invalidate_caches()
            global torch
            torch = importlib.import_module("torch")
        except Exception:
            logger.exception("PyTorch installation failed")
            raise RuntimeError("Failed to install MPS-enabled PyTorch")
        return configure_gpu_env(_attempt_install=False)

    try:
        system = platform.system()
        if system == "Darwin":
            machine = platform.machine()
            if machine == "x86_64":
                logger.warning("Running under Rosetta (x86_64); falling back to CPU")
            else:
                install_hint = (
                    "Install with: pip install torch==2.1.0 torchvision==0.16.0 "
                    "--extra-index-url https://download.pytorch.org/whl/metal"
                )
                mps_backend = getattr(torch.backends, "mps", None)
                if not mps_backend:
                    if _attempt_install:
                        return _install_and_retry("MPS backend not present")
                    logger.warning(
                        "MPS backend not present; falling back to CPU. %s", install_hint
                    )
                elif not torch.backends.mps.is_built():
                    if _attempt_install:
                        return _install_and_retry("MPS backend not built")
                    logger.warning(
                        "MPS backend not built; falling back to CPU. %s", install_hint
                    )
                elif not torch.backends.mps.is_available():
                    if _attempt_install:
                        return _install_and_retry("MPS backend not available")
                    logger.warning(
                        "MPS backend not available; falling back to CPU. %s", install_hint
                    )
                else:
                    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                    try:
                        torch.ones(1, device="mps").cpu()
                        env["TORCH_DEVICE"] = "mps"
                        os.environ.update(env)
                        return True
                    except Exception:
                        logger.exception("Tensor operation failed on mps backend")
                        if _attempt_install:
                            return _install_and_retry(
                                "Tensor operation failed on mps backend"
                            )
        elif torch.cuda.is_available():
            try:
                torch.ones(1, device="cuda").cpu()
                env["TORCH_DEVICE"] = "cuda"
                os.environ.update(env)
                return True
            except Exception:
                logger.exception("Tensor operation failed on cuda backend")
        else:
            logger.warning("CUDA backend not available; falling back to CPU")
    except Exception:
        logger.exception("Exception during GPU detection")

    logger.warning("No GPU backend detected; falling back to CPU")
    env.setdefault("TORCH_DEVICE", "cpu")
    os.environ.update(env)
    return False



def get_gpu_backend() -> str | None:
    """Return the default GPU backend name if available."""

    if torch is not None:
        try:  # pragma: no cover - optional dependency
            mps_built = hasattr(torch, "backends") and hasattr(torch.backends, "mps")
            mps_available = mps_built and torch.backends.mps.is_available()
            if mps_built and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
                logging.getLogger(__name__).warning(
                    "MPS support detected but PYTORCH_ENABLE_MPS_FALLBACK is not set to '1'. "
                    "Export PYTORCH_ENABLE_MPS_FALLBACK=1 to enable CPU fallback for unsupported ops."
                )
            if mps_built:
                os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            if torch.cuda.is_available() or mps_available:
                return "torch"
        except Exception:
            pass
    try:  # pragma: no cover - optional dependency
        import cupy as cp  # type: ignore

        if cp.cuda.runtime.getDeviceCount() > 0:
            return "cupy"
    except Exception:
        pass
    return None


def get_default_device(device: str | "torch.device" | None = "auto") -> "torch.device":
    """Return a valid :class:`torch.device`.

    Parameters
    ----------
    device:
        Desired device identifier. If ``"auto"`` or ``None`` the function prefers
        Apple's Metal backend (``"mps"``) on macOS when available, then
        ``"cuda"`` or finally ``"cpu"``. If a non-CPU device is requested but
        unavailable the call falls back to the CPU.
    """

    if torch is None:
        raise RuntimeError("PyTorch is required for device selection")
    if device is None or (isinstance(device, str) and device == "auto"):
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if isinstance(device, str) and device not in ("cpu", "mps") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device) if isinstance(device, str) else device


def _main() -> int:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-gpu", action="store_true", help="exit 0 if a GPU is available")
    parser.add_argument(
        "--setup-env",
        action="store_true",
        help="print export commands configuring GPU environment variables",
    )
    args = parser.parse_args()
    if args.check_gpu:
        return 0 if configure_gpu_env() else 1
    if args.setup_env:
        configure_gpu_env()
        for k in ("TORCH_DEVICE", "PYTORCH_ENABLE_MPS_FALLBACK"):
            v = os.environ.get(k)
            if v is not None:
                print(f"export {k}={v}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(_main())
