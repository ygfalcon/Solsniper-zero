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


def ensure_torch_with_metal() -> None:
    """Ensure PyTorch with Metal backend is installed on macOS arm64.

    The helper installs specific ``torch`` and ``torchvision`` versions from the
    Metal wheels when running on Apple Silicon.  After installation the module is
    imported and the ``mps`` backend availability is verified.  A ``RuntimeError``
    is raised if the backend remains unavailable.
    """

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return

    extra_index = ["--extra-index-url", "https://download.pytorch.org/whl/metal"]
    logger = logging.getLogger(__name__)

    global torch
    try:
        if (
            torch is not None
            and sys.modules.get("torch") is torch
            and getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            return
    except Exception:
        pass

    from scripts.startup import _pip_install

    try:
        _pip_install("torch==2.1.0", "torchvision==0.16.0", *extra_index)
    except SystemExit as exc:  # pragma: no cover - installation failure
        logger.exception("PyTorch installation failed")
        raise RuntimeError("Failed to install MPS-enabled PyTorch") from exc

    importlib.invalidate_caches()
    torch = importlib.import_module("torch")

    if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
        logger.warning("MPS backend not available; attempting to reinstall Metal wheel")
        try:
            _pip_install(
                "--force-reinstall",
                "torch==2.1.0",
                "torchvision==0.16.0",
                *extra_index,
            )
        except SystemExit as exc:  # pragma: no cover - installation failure
            logger.exception("PyTorch reinstallation failed")
            raise RuntimeError("Failed to reinstall MPS-enabled PyTorch") from exc
        importlib.invalidate_caches()
        torch = importlib.reload(torch)
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS backend still not available. install the Metal wheel manually with: pip install "
                "torch==2.1.0 torchvision==0.16.0 --extra-index-url "
                "https://download.pytorch.org/whl/metal",
            )


def detect_gpu(_attempt_install: bool = True) -> bool:
    """Return ``True`` when a supported GPU backend is available.

    The check prefers Apple's Metal backend (MPS) on macOS machines with
    Apple Silicon and falls back to CUDA on other platforms.  After the
    usual availability checks a tiny tensor is created and moved back to
    the CPU to ensure the backend is operational.  Any import errors,
    unsupported configurations or runtime failures are treated as absence
    of a GPU.
    """

    if torch is None:
        if _attempt_install:
            try:
                ensure_torch_with_metal()
            except Exception:
                logging.getLogger(__name__).exception(
                    "PyTorch installation failed; GPU unavailable"
                )
                return False
            return detect_gpu(_attempt_install=False)
        logging.getLogger(__name__).warning(
            "PyTorch is not installed; GPU unavailable"
        )
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
            install_hint = (
                "Install with: pip install torch==2.1.0 torchvision==0.16.0 "
                "--extra-index-url https://download.pytorch.org/whl/metal"
            )

            def _install_and_retry(reason: str) -> bool:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "%s; attempting to install MPS-enabled PyTorch", reason
                )
                try:
                    ensure_torch_with_metal()
                except Exception:
                    logger.exception("PyTorch installation failed")
                    raise RuntimeError(
                        "Failed to install MPS-enabled PyTorch"
                    )
                if detect_gpu(_attempt_install=False):
                    logger.info("MPS backend detected after installation")
                    return True
                logger.error("MPS backend unavailable after installation")
                raise RuntimeError(
                    "MPS backend unavailable even after installing PyTorch"
                )

            if not getattr(torch.backends, "mps", None):
                if _attempt_install:
                    return _install_and_retry("MPS backend not present")
                logging.getLogger(__name__).warning(
                    "MPS backend not present; GPU unavailable. %s", install_hint
                )
                return False
            if not torch.backends.mps.is_built():
                if _attempt_install:
                    return _install_and_retry("MPS backend not built")
                logging.getLogger(__name__).warning(
                    "MPS backend not built; GPU unavailable. %s", install_hint
                )
                return False
            if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            elif os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
                logging.getLogger(__name__).warning(
                    "PYTORCH_ENABLE_MPS_FALLBACK is not set to '1'; GPU unavailable",
                )
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            if not torch.backends.mps.is_available():
                if _attempt_install:
                    return _install_and_retry("MPS backend not available")
                logging.getLogger(__name__).warning(
                    "MPS backend not available; GPU unavailable. %s", install_hint
                )
                return False
            try:
                torch.ones(1, device="mps").cpu()
            except Exception:
                logging.getLogger(__name__).exception(
                    "Tensor operation failed on mps backend"
                )
                if _attempt_install:
                    return _install_and_retry(
                        "Tensor operation failed on mps backend"
                    )
                return False
            return True
        if not torch.cuda.is_available():
            logging.getLogger(__name__).warning("CUDA backend not available")
            return False
        try:
            torch.ones(1, device="cuda").cpu()
        except Exception:
            logging.getLogger(__name__).exception(
                "Tensor operation failed on cuda backend"
            )
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


def ensure_gpu_env() -> dict[str, str]:
    """Configure environment variables for GPU execution.

    If a GPU backend is available, ``TORCH_DEVICE`` is set to the preferred
    device (``"mps"`` on macOS with Apple Silicon or ``"cuda"`` elsewhere).
    When the Metal backend is used, ``PYTORCH_ENABLE_MPS_FALLBACK`` is set to
    ``"1"`` to allow CPU fallback for unsupported operations.  The function
    returns a dictionary of variables that were modified.
    """

    env: dict[str, str] = {}
    if torch is None:
        return env
    try:
        system = platform.system()
        if system == "Darwin" and torch.backends.mps.is_available():
            env["TORCH_DEVICE"] = "mps"
            env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        elif torch.cuda.is_available():
            env["TORCH_DEVICE"] = "cuda"
        for key, value in env.items():
            os.environ[key] = value
        return env
    except Exception:  # pragma: no cover - best effort only
        logging.getLogger(__name__).exception("Exception during GPU env setup")
        return env


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
        return 0 if detect_gpu() else 1
    if args.setup_env:
        for k, v in ensure_gpu_env().items():
            print(f"export {k}={v}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(_main())
