from __future__ import annotations

import argparse
import logging
import os
import platform

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch is optional at runtime
    torch = None  # type: ignore


def detect_gpu() -> bool:
    """Return ``True`` when a supported GPU backend is available.

    The check prefers Apple's Metal backend (MPS) on macOS machines with
    Apple Silicon and falls back to CUDA on other platforms.  Any import
    errors or unsupported configurations are treated as absence of a GPU.
    """

    if torch is None:
        return False
    try:
        if platform.system() == "Darwin":
            return bool(torch.backends.mps.is_available())
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def get_default_backend() -> str | None:
    """Return the default GPU backend name if available."""

    if torch is not None:
        try:  # pragma: no cover - optional dependency
            mps_available = (
                hasattr(torch, "backends")
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
            if torch.cuda.is_available() or mps_available:
                if mps_available and os.environ.setdefault(
                    "PYTORCH_ENABLE_MPS_FALLBACK", "1"
                ) != "1":
                    logging.getLogger(__name__).warning(
                        "MPS is available but PYTORCH_ENABLE_MPS_FALLBACK is not set to '1'"
                    )
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


def select_device(device: str | "torch.device" | None = "auto") -> "torch.device":
    """Return a valid :class:`torch.device`.

    Parameters
    ----------
    device:
        Desired device identifier. If ``"auto"`` or ``None`` the function
        chooses ``"cuda"`` or ``"mps"`` when available, otherwise ``"cpu"``.
        If a non-CPU device is requested but unavailable the call falls back
        to the CPU.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for device selection")
    if device is None or (isinstance(device, str) and device == "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if isinstance(device, str) and device not in ("cpu", "mps") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device) if isinstance(device, str) else device


def get_default_device() -> "torch.device":
    """Return the preferred accelerator device if available."""
    if torch is None:
        raise RuntimeError("PyTorch is required for device selection")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # pragma: no cover - optional backend
        return torch.device("mps")
    return torch.device("cpu")


def _main() -> int:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-gpu", action="store_true", help="exit 0 if a GPU is available")
    args = parser.parse_args()
    if args.check_gpu:
        return 0 if detect_gpu() else 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(_main())
