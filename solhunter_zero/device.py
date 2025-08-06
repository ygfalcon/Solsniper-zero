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
        logging.getLogger(__name__).warning("PyTorch is not installed; GPU unavailable")
        return False
    try:
        system = platform.system()
        if system == "Darwin":
            machine = platform.machine()
            if machine == "x86_64":
                logging.getLogger(__name__).warning(
                    "Running under Rosetta (x86_64); GPU unavailable"
                )
                return False
            if not getattr(torch.backends, "mps", None):
                logging.getLogger(__name__).warning(
                    "MPS backend not present; GPU unavailable"
                )
                return False
            if not torch.backends.mps.is_built():
                logging.getLogger(__name__).warning(
                    "MPS backend not built; GPU unavailable"
                )
                return False
            if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
                logging.getLogger(__name__).warning(
                    "PYTORCH_ENABLE_MPS_FALLBACK is not set to '1'; GPU unavailable"
                )
                return False
            if not torch.backends.mps.is_available():
                logging.getLogger(__name__).warning(
                    "MPS backend not available"
                )
                return False
            return True
        if not torch.cuda.is_available():
            logging.getLogger(__name__).warning("CUDA backend not available")
            return False
        return True
    except Exception:
        logging.getLogger(__name__).exception("Exception during GPU detection")
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
    args = parser.parse_args()
    if args.check_gpu:
        return 0 if detect_gpu() else 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(_main())
