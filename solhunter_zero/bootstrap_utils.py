from __future__ import annotations

import importlib.util
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from scripts import deps

from . import device
from .device import METAL_EXTRA_INDEX
from .logging_utils import log_startup
from .paths import ROOT

DEPS_MARKER = ROOT / ".cache" / "deps-installed"


@dataclass
class DepsConfig:
    """Configuration for :func:`ensure_deps`."""

    install_optional: bool = False
    extras: Sequence[str] | None = ("uvloop",)
    ensure_wallet_cli: bool = True


def ensure_venv(argv: list[str] | None) -> None:
    """Create a local virtual environment and re-invoke the script inside it."""
    if argv is not None:
        return

    venv_dir = ROOT / ".venv"
    python = (
        venv_dir
        / ("Scripts" if os.name == "nt" else "bin")
        / ("python.exe" if os.name == "nt" else "python")
    )

    if venv_dir.exists():
        if not python.exists() or not os.access(str(python), os.X_OK):
            print("Virtual environment missing interpreter; recreating .venv...")
            shutil.rmtree(venv_dir)

    if not venv_dir.exists():
        print("Creating virtual environment in .venv...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"Failed to create .venv: {exc}")
            raise SystemExit(1)

    def _inspect(p: Path) -> tuple[str | None, tuple[int, int, int]]:
        try:
            info = subprocess.check_output(
                [
                    str(p),
                    "-c",
                    (
                        "import json, platform, sys;"
                        "print(json.dumps({'machine': platform.machine(), 'version': sys.version_info[:3]}))"
                    ),
                ],
                text=True,
            )
            data = json.loads(info)
            return data.get("machine"), tuple(data.get("version", []))
        except Exception as exc:  # pragma: no cover - hard failure
            print(f"Failed to inspect virtual environment interpreter: {exc}")
            return None, (0, 0, 0)

    machine, version = _inspect(python)

    if platform.system() == "Darwin" and (machine != "arm64" or version < (3, 11)):
        brew_python = shutil.which("python3.11")
        if not brew_python:
            print(
                "python3.11 from Homebrew not found. "
                "Install it with 'brew install python@3.11'."
            )
            raise SystemExit(1)
        print("Recreating .venv using Homebrew python3.11...")
        shutil.rmtree(venv_dir, ignore_errors=True)
        try:
            subprocess.check_call([brew_python, "-m", "venv", str(venv_dir)])
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"Failed to create .venv with Homebrew python3.11: {exc}")
            raise SystemExit(1)
        python = (
            venv_dir
            / ("Scripts" if os.name == "nt" else "bin")
            / ("python.exe" if os.name == "nt" else "python")
        )
        machine, version = _inspect(python)

    if version < (3, 11):
        print("Recreating .venv using current interpreter...")
        shutil.rmtree(venv_dir, ignore_errors=True)
        try:
            subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"Failed to create .venv with current interpreter: {exc}")
            raise SystemExit(1)
        python = (
            venv_dir
            / ("Scripts" if os.name == "nt" else "bin")
            / ("python.exe" if os.name == "nt" else "python")
        )

    if Path(sys.prefix) != venv_dir:
        try:
            os.execv(str(python), [str(python), *sys.argv])
            raise RuntimeError("exec failed")
        except OSError as exc:
            msg = f"Failed to execv {python}: {exc}"
            logging.exception(msg)
            log_startup(msg)
            raise


def _pip_install(*args: str, retries: int = 3) -> None:
    """Run ``pip install`` with retries and exponential backoff."""
    errors: list[str] = []
    cmd: list[str] = []
    for attempt in range(1, retries + 1):
        cmd = [sys.executable, "-m", "pip", "install", *args]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        log_startup(f"{' '.join(cmd)} (attempt {attempt}/{retries})")
        if result.stdout:
            log_startup(result.stdout.rstrip())
        if result.stderr:
            log_startup(result.stderr.rstrip())
        if result.returncode == 0:
            log_startup(f"pip install {' '.join(args)} succeeded on attempt {attempt}")
            return
        errors.append(result.stderr.strip() or result.stdout.strip())
        if attempt < retries:
            wait = 2 ** (attempt - 1)
            print(
                f"pip install {' '.join(args)} failed (attempt {attempt}/{retries}). Retrying in {wait} seconds..."
            )
            time.sleep(wait)
    msg = f"Failed to install {' '.join(args)} after {retries} attempts"
    retry_cmd = " ".join(cmd)
    print(f"{msg}. To retry manually, run: {retry_cmd}")
    log_startup(
        json.dumps(
            {
                "event": "pip_install_failed",
                "cmd": retry_cmd,
                "errors": [e for e in errors if e],
            }
        )
    )
    raise SystemExit(result.returncode)


def _package_missing(pkg: str) -> bool:
    """Return ``True`` if *pkg* is not installed.

    The check uses ``pip show`` which is fast and avoids unnecessary
    installations.  When a package is already present a message is logged to
    ``startup.log`` for transparency.
    """

    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", pkg],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        log_startup(f"Skipping installation of {pkg}; already satisfied")
        return False
    return True


def ensure_deps(
    cfg: DepsConfig | None = None,
    *,
    install_optional: bool = False,
    extras: Sequence[str] | None = ("uvloop",),
    ensure_wallet_cli: bool = True,
    full: bool | None = None,
) -> None:
    """Ensure Python dependencies and optional extras are installed.

    Parameters
    ----------
    cfg:
        Optional :class:`DepsConfig` describing installation behaviour. When
        omitted, legacy keyword arguments are used to build a configuration.
    install_optional, extras, ensure_wallet_cli:
        Deprecated keyword arguments retained for backward compatibility when
        *cfg* is not provided.
    full:
        When ``True`` install optional dependencies. Deprecated alias for
        ``install_optional``.
    """

    if full is not None:
        install_optional = full

    if cfg is None:
        cfg = DepsConfig(
            install_optional=install_optional,
            extras=extras,
            ensure_wallet_cli=ensure_wallet_cli,
        )
    elif full is not None:
        cfg.install_optional = full

    if platform.system() == "Darwin":
        from . import macos_setup

        if not macos_setup.mac_setup_completed():
            report = macos_setup.prepare_macos_env(non_interactive=True)
            if not report.get("success"):
                for step, info in report.get("steps", {}).items():
                    if info.get("status") == "error":
                        fix = macos_setup.MANUAL_FIXES.get(step)
                        if fix:
                            print(f"Manual fix for {step}: {fix}")
                print(
                    "macOS environment preparation failed. Please address the issues above and re-run.",
                )
                raise SystemExit(1)

    force_reinstall = os.getenv("SOLHUNTER_FORCE_DEPS") == "1"
    if DEPS_MARKER.exists() and not force_reinstall:
        log_startup("Dependency marker present; skipping installation")
        from . import bootstrap as bootstrap_mod

        bootstrap_mod.ensure_route_ffi()
        bootstrap_mod.ensure_depth_service()
        return

    req, opt = deps.check_deps()
    if req:
        print("Missing required modules: " + ", ".join(req))
    if opt:
        print(
            "Optional modules missing: " + ", ".join(opt) + " (features disabled).",
        )

    # Filter out packages that are already satisfied according to pip.
    req = [m for m in req if _package_missing(m.replace("_", "-"))]
    if cfg.install_optional:
        opt = [m for m in opt if _package_missing(m.replace("_", "-"))]
    else:
        opt = []

    need_cli = cfg.ensure_wallet_cli and shutil.which("solhunter-wallet") is None
    need_install = bool(req) or need_cli or (cfg.install_optional and opt)
    if not need_install:
        from . import bootstrap as bootstrap_mod

        bootstrap_mod.ensure_route_ffi()
        bootstrap_mod.ensure_depth_service()
        DEPS_MARKER.parent.mkdir(parents=True, exist_ok=True)
        DEPS_MARKER.write_text(
            json.dumps(
                {
                    "extras": list(cfg.extras) if cfg.extras else [],
                    "install_optional": cfg.install_optional,
                    "timestamp": time.time(),
                }
            )
        )
        return

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
        text=True,
    )
    if pip_check.returncode != 0:
        if "No module named pip" in pip_check.stderr:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "ensurepip", "--default-pip"]
                )
            except (
                subprocess.CalledProcessError
            ) as exc:  # pragma: no cover - hard failure
                print(f"Failed to bootstrap pip: {exc}")
                raise SystemExit(exc.returncode)
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "--version"])
        else:  # pragma: no cover - unexpected failure
            print(f"Failed to invoke pip: {pip_check.stderr.strip()}")
            raise SystemExit(pip_check.returncode)

    extra_index: list[str] = []
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        extra_index = list(METAL_EXTRA_INDEX)

    need_install = (
        bool(req) or need_cli or (cfg.install_optional and (opt or extra_index))
    )
    if need_install:
        import contextlib
        import io

        from scripts import preflight

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            try:
                preflight.check_internet()
            except SystemExit as exc:
                raise SystemExit(
                    "Unable to establish an internet connection; aborting."
                ) from exc

    if req or need_cli:
        print("Installing required dependencies...")
        extras_arg = ""
        if cfg.extras:
            extras_arg = f".[{','.join(cfg.extras)}]"
        else:
            extras_arg = "."
        _pip_install(extras_arg, *extra_index)
        failed_req = [m for m in req if importlib.util.find_spec(m) is None]
        if failed_req:
            print(
                "Missing required dependencies after installation: "
                + ", ".join(failed_req)
            )
            raise SystemExit(1)
        if need_cli and shutil.which("solhunter-wallet") is None:
            print(
                "'solhunter-wallet' still not available after installation. Aborting."
            )
            raise SystemExit(1)

    if cfg.install_optional and extra_index:
        try:
            device.initialize_gpu()
        except Exception as exc:
            print(str(exc))
            raise SystemExit(str(exc))
        if "torch" in opt:
            opt.remove("torch")

    if cfg.install_optional and opt:
        print("Installing optional dependencies...")
        mapping = {
            "faiss": "faiss-cpu",
            "sentence_transformers": "sentence-transformers",
            "torch": "torch",
            "orjson": "orjson",
            "lz4": "lz4",
            "zstandard": "zstandard",
            "msgpack": "msgpack",
        }
        mods = set(opt)
        extras_pkgs: list[str] = []
        if "orjson" in mods and _package_missing("fastjson"):
            extras_pkgs.append("fastjson")
        if {"lz4", "zstandard"} & mods and _package_missing("fastcompress"):
            extras_pkgs.append("fastcompress")
        if "msgpack" in mods and _package_missing("msgpack"):
            extras_pkgs.append("msgpack")
        if extras_pkgs:
            _pip_install(f".[{','.join(extras_pkgs)}]", *extra_index)
        remaining = mods - {"orjson", "lz4", "zstandard", "msgpack"}
        for name in remaining:
            pkg = mapping.get(name, name.replace("_", "-"))
            if _package_missing(pkg):
                _pip_install(pkg, *extra_index)

        missing_opt = [m for m in opt if importlib.util.find_spec(m) is None]
        if missing_opt:
            print(
                "Optional modules missing: "
                + ", ".join(missing_opt)
                + " (features disabled)."
            )

    from . import bootstrap as bootstrap_mod

    bootstrap_mod.ensure_route_ffi()
    bootstrap_mod.ensure_depth_service()

    DEPS_MARKER.parent.mkdir(parents=True, exist_ok=True)
    DEPS_MARKER.write_text(
        json.dumps(
            {
                "extras": list(cfg.extras) if cfg.extras else [],
                "install_optional": cfg.install_optional,
                "timestamp": time.time(),
            }
        )
    )


def ensure_endpoints(cfg: dict) -> None:
    """Ensure HTTP endpoints in ``cfg`` are reachable.

    The configuration may specify several service URLs such as
    ``DEX_BASE_URL`` or custom metrics endpoints. This function attempts a
    ``HEAD`` request to each HTTP(S) URL and aborts startup if any service is
    unreachable. BirdEye is only checked when an API key is configured.
    """

    import asyncio
    import urllib.error

    from solhunter_zero.http import check_endpoint

    urls: dict[str, str] = {}
    if cfg.get("birdeye_api_key"):
        urls["BirdEye"] = "https://public-api.birdeye.so/defi/tokenlist"
    for key, val in cfg.items():
        if not isinstance(val, str):
            continue
        if not val.startswith("http://") and not val.startswith("https://"):
            continue
        urls[key] = val

    async def _check(name: str, url: str) -> tuple[str, Exception] | None:
        # Each URL is checked with its own exponential backoff.
        for attempt in range(3):
            try:
                # ``check_endpoint`` is synchronous; run it in a thread to avoid blocking.
                await asyncio.to_thread(check_endpoint, url, retries=1)
                return None
            except urllib.error.URLError as exc:  # pragma: no cover - network failure
                if attempt == 2:
                    return name, exc
                wait = 2**attempt
                print(
                    f"Attempt {attempt + 1} failed for {name} at {url}: {exc}. "
                    f"Retrying in {wait} seconds..."
                )
                await asyncio.sleep(wait)

    async def _run() -> list[tuple[str, Exception] | None]:
        tasks = [_check(name, url) for name, url in urls.items()]
        return await asyncio.gather(*tasks)

    results = asyncio.run(_run())
    failures: list[tuple[str, str, Exception]] = []
    for name_exc in results:
        if name_exc is None:
            continue
        name, exc = name_exc
        url = urls[name]
        failures.append((name, url, exc))

    if failures:
        details = "; ".join(f"{name} at {url} ({exc})" for name, url, exc in failures)
        print(
            "Failed to reach the following endpoints: "
            f"{details}. Check your network connection or configuration."
        )
        raise SystemExit(1)


def _run_rustup_setup(cmd, *, shell: bool = False, retries: int = 2) -> None:
    """Run ``cmd`` retrying on failure with helpful errors."""

    for attempt in range(1, retries + 1):
        try:
            subprocess.check_call(cmd, shell=shell)
            return
        except subprocess.CalledProcessError as exc:
            if attempt == retries:
                print(
                    "Failed to install Rust toolchain via rustup. "
                    "Please visit https://rustup.rs/ and follow the instructions.",
                )
                raise SystemExit(exc.returncode)
            time.sleep(1)
            print("Rustup setup failed, retrying...")


def ensure_cargo() -> None:
    installed = False
    cache_marker = ROOT / ".cache" / "cargo-installed"
    if platform.system() == "Darwin":
        from solhunter_zero.macos_setup import apply_brew_env, ensure_tools

        ensure_tools()
        apply_brew_env()

    cargo_bin = Path.home() / ".cargo" / "bin"
    os.environ["PATH"] = f"{cargo_bin}{os.pathsep}{os.environ.get('PATH', '')}"

    if shutil.which("cargo") is None:
        if cache_marker.exists():
            print(
                "Rust toolchain previously installed but 'cargo' was not found. "
                "Ensure ~/.cargo/bin is in your PATH or remove the cache marker and rerun the script.",
            )
            raise SystemExit(1)

        cache_marker.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("brew") is not None:
            print("Installing rustup with Homebrew...")
            try:
                subprocess.check_call(["brew", "install", "rustup"])
            except subprocess.CalledProcessError as exc:
                print(f"Homebrew failed to install rustup: {exc}")
                raise SystemExit(exc.returncode)
            _run_rustup_setup(["rustup-init", "-y"])
        else:
            if shutil.which("curl") is None:
                print(
                    "curl is required to install the Rust toolchain. "
                    "Install it (e.g., with Homebrew: 'brew install curl') and re-run this script.",
                )
                raise SystemExit(1)
            print("Installing Rust toolchain via rustup...")
            _run_rustup_setup(
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
                shell=True,
            )
        installed = True
        cache_marker.write_text("ok")

    try:
        subprocess.check_call(["cargo", "--version"], stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        print("Failed to run 'cargo --version'. Is Rust installed correctly?")
        raise SystemExit(exc.returncode)
    if installed and platform.system() == "Darwin" and platform.machine() == "arm64":
        subprocess.check_call(["rustup", "target", "add", "aarch64-apple-darwin"])
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            targets = subprocess.check_output(["rustup", "target", "list"], text=True)
        except subprocess.CalledProcessError as exc:
            print("Failed to list rust targets. Is rustup installed correctly?")
            raise SystemExit(exc.returncode)
        if "aarch64-apple-darwin" not in targets:
            subprocess.check_call(["rustup", "target", "add", "aarch64-apple-darwin"])

    missing = [tool for tool in ("pkg-config", "cmake") if shutil.which(tool) is None]
    if missing:
        if platform.system() == "Darwin" and shutil.which("brew") is not None:
            print(
                f"Missing {', '.join(missing)}. Attempting to install with Homebrew...",
            )
            try:
                subprocess.check_call(["brew", "install", "pkg-config", "cmake"])
            except subprocess.CalledProcessError as exc:
                print(f"Homebrew installation failed: {exc}")
            else:
                missing = [
                    tool
                    for tool in ("pkg-config", "cmake")
                    if shutil.which(tool) is None
                ]
        if missing:
            names = " and ".join(missing)
            brew = " ".join(missing)
            print(
                f"{names} {'are' if len(missing) > 1 else 'is'} required to build native extensions. "
                f"Install {'them' if len(missing) > 1 else 'it'} (e.g., with Homebrew: 'brew install {brew}') and re-run this script.",
            )
            raise SystemExit(1)
