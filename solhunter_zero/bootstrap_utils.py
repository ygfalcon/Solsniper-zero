from __future__ import annotations

import json
import importlib.util
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from typing import Sequence

from scripts import deps
from . import device
from .device import METAL_EXTRA_INDEX
from .logging_utils import log_step

ROOT = Path(__file__).resolve().parent.parent


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
            log_step("Virtual environment missing interpreter; recreating .venv...")
            shutil.rmtree(venv_dir)

    if not venv_dir.exists():
        log_step("Creating virtual environment in .venv...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        except (subprocess.CalledProcessError, OSError) as exc:
            log_step(f"Failed to create .venv: {exc}")
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
            log_step(f"Failed to inspect virtual environment interpreter: {exc}")
            return None, (0, 0, 0)

    machine, version = _inspect(python)

    if platform.system() == "Darwin" and (machine != "arm64" or version < (3, 11)):
        brew_python = shutil.which("python3.11")
        if not brew_python:
            log_step(
                "python3.11 from Homebrew not found. Install it with 'brew install python@3.11'."
            )
            raise SystemExit(1)
        log_step("Recreating .venv using Homebrew python3.11...")
        shutil.rmtree(venv_dir, ignore_errors=True)
        try:
            subprocess.check_call([brew_python, "-m", "venv", str(venv_dir)])
        except (subprocess.CalledProcessError, OSError) as exc:
            log_step(f"Failed to create .venv with Homebrew python3.11: {exc}")
            raise SystemExit(1)
        python = (
            venv_dir
            / ("Scripts" if os.name == "nt" else "bin")
            / ("python.exe" if os.name == "nt" else "python")
        )
        machine, version = _inspect(python)

    if version < (3, 11):
        log_step("Recreating .venv using current interpreter...")
        shutil.rmtree(venv_dir, ignore_errors=True)
        try:
            subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        except (subprocess.CalledProcessError, OSError) as exc:
            log_step(f"Failed to create .venv with current interpreter: {exc}")
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
            log_step(msg)
            raise


def _pip_install(*args: str, retries: int = 3) -> None:
    """Run ``pip install`` with retries and exponential backoff."""
    errors: list[str] = []
    for attempt in range(1, retries + 1):
        cmd = [sys.executable, "-m", "pip", "install", *args]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        log_step(f"{' '.join(cmd)} (attempt {attempt}/{retries})")
        if result.stdout:
            log_step(result.stdout.rstrip())
        if result.stderr:
            log_step(result.stderr.rstrip())
        if result.returncode == 0:
            log_step(
                f"pip install {' '.join(args)} succeeded on attempt {attempt}"
            )
            return
        errors.append(result.stderr.strip() or result.stdout.strip())
        if attempt < retries:
            wait = 2 ** (attempt - 1)
            log_step(
                f"pip install {' '.join(args)} failed (attempt {attempt}/{retries}). Retrying in {wait} seconds..."
            )
            time.sleep(wait)
    msg = f"Failed to install {' '.join(args)} after {retries} attempts:"
    log_step(msg)
    for err in errors:
        if err:
            log_step(err)
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
        log_step(f"Skipping installation of {pkg}; already satisfied")
        return False
    return True


def ensure_deps(
    *,
    install_optional: bool = False,
    extras: Sequence[str] | None = ("uvloop",),
    ensure_wallet_cli: bool = True,
) -> None:
    """Ensure Python dependencies and optional extras are installed.

    Parameters
    ----------
    install_optional:
        Install optional modules defined in :mod:`scripts.deps` when True.
    extras:
        Iterable of ``pyproject.toml`` extras to install when the local package
        itself must be installed.  Defaults to ``("uvloop",)``.
    ensure_wallet_cli:
        When ``True`` the ``solhunter-wallet`` command is ensured to be
        available by installing the current package if necessary.
    """

    if platform.system() == "Darwin":
        from . import macos_setup

        report = macos_setup.prepare_macos_env(non_interactive=True)
        if not report.get("success"):
            for step, info in report.get("steps", {}).items():
                if info.get("status") == "error":
                    fix = macos_setup.MANUAL_FIXES.get(step)
                    if fix:
                        log_step(f"Manual fix for {step}: {fix}")
            log_step(
                "macOS environment preparation failed. Please address the issues above and re-run."
            )
            raise SystemExit(1)

    req, opt = deps.check_deps()
    if req:
        log_step("Missing required modules: " + ", ".join(req))
    if opt:
        log_step(
            "Optional modules missing: " + ", ".join(opt) + " (features disabled)."
        )

    # Filter out packages that are already satisfied according to pip.
    req = [m for m in req if _package_missing(m.replace("_", "-"))]
    if install_optional:
        opt = [m for m in opt if _package_missing(m.replace("_", "-"))]
    else:
        opt = []

    need_cli = ensure_wallet_cli and shutil.which("solhunter-wallet") is None
    need_install = bool(req) or need_cli or (install_optional and opt)
    if not need_install:
        from . import bootstrap as bootstrap_mod

        bootstrap_mod.ensure_route_ffi()
        bootstrap_mod.ensure_depth_service()
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
            except subprocess.CalledProcessError as exc:  # pragma: no cover - hard failure
                log_step(f"Failed to bootstrap pip: {exc}")
                raise SystemExit(exc.returncode)
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "--version"])
        else:  # pragma: no cover - unexpected failure
            log_step(f"Failed to invoke pip: {pip_check.stderr.strip()}")
            raise SystemExit(pip_check.returncode)

    extra_index: list[str] = []
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        extra_index = list(METAL_EXTRA_INDEX)

    need_install = bool(req) or need_cli or (install_optional and (opt or extra_index))
    if need_install:
        from scripts import startup
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            try:
                startup.check_internet()
            except SystemExit as exc:
                raise SystemExit(
                    "Unable to establish an internet connection; aborting."
                ) from exc

    if req or need_cli:
        log_step("Installing required dependencies...")
        extras_arg = ""
        if extras:
            extras_arg = f".[{','.join(extras)}]"
        else:
            extras_arg = "."
        _pip_install(extras_arg, *extra_index)
        failed_req = [m for m in req if importlib.util.find_spec(m) is None]
        if failed_req:
            log_step(
                "Missing required dependencies after installation: "
                + ", ".join(failed_req)
            )
            raise SystemExit(1)
        if need_cli and shutil.which("solhunter-wallet") is None:
            log_step("'solhunter-wallet' still not available after installation. Aborting.")
            raise SystemExit(1)

    if install_optional and extra_index:
        try:
            device.initialize_gpu()
        except Exception as exc:
            log_step(str(exc))
            raise SystemExit(str(exc))
        if "torch" in opt:
            opt.remove("torch")

    if install_optional and opt:
        log_step("Installing optional dependencies...")
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
            log_step(
                "Optional modules missing: "
                + ", ".join(missing_opt)
                + " (features disabled)."
            )

    from . import bootstrap as bootstrap_mod
    bootstrap_mod.ensure_route_ffi()
    bootstrap_mod.ensure_depth_service()


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
                log_step(
                    f"Attempt {attempt + 1} failed for {name} at {url}: {exc}. Retrying in {wait} seconds..."
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
        details = "; ".join(
            f"{name} at {url} ({exc})" for name, url, exc in failures
        )
        log_step(
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
                log_step(
                    "Failed to install Rust toolchain via rustup. Please visit https://rustup.rs/ and follow the instructions."
                )
                raise SystemExit(exc.returncode)
            time.sleep(1)
            log_step("Rustup setup failed, retrying...")


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
            log_step(
                "Rust toolchain previously installed but 'cargo' was not found. Ensure ~/.cargo/bin is in your PATH or remove the cache marker and rerun the script."
            )
            raise SystemExit(1)

        cache_marker.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("brew") is not None:
            log_step("Installing rustup with Homebrew...")
            try:
                subprocess.check_call(["brew", "install", "rustup"])
            except subprocess.CalledProcessError as exc:
                log_step(f"Homebrew failed to install rustup: {exc}")
                raise SystemExit(exc.returncode)
            _run_rustup_setup(["rustup-init", "-y"])
        else:
            if shutil.which("curl") is None:
                log_step(
                    "curl is required to install the Rust toolchain. Install it (e.g., with Homebrew: 'brew install curl') and re-run this script."
                )
                raise SystemExit(1)
            log_step("Installing Rust toolchain via rustup...")
            _run_rustup_setup(
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
                shell=True,
            )
        installed = True
        cache_marker.write_text("ok")

    try:
        subprocess.check_call(["cargo", "--version"], stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        log_step("Failed to run 'cargo --version'. Is Rust installed correctly?")
        raise SystemExit(exc.returncode)
    if installed and platform.system() == "Darwin" and platform.machine() == "arm64":
        subprocess.check_call(["rustup", "target", "add", "aarch64-apple-darwin"])
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            targets = subprocess.check_output(["rustup", "target", "list"], text=True)
        except subprocess.CalledProcessError as exc:
            log_step("Failed to list rust targets. Is rustup installed correctly?")
            raise SystemExit(exc.returncode)
        if "aarch64-apple-darwin" not in targets:
            subprocess.check_call(["rustup", "target", "add", "aarch64-apple-darwin"])

    missing = [tool for tool in ("pkg-config", "cmake") if shutil.which(tool) is None]
    if missing:
        if platform.system() == "Darwin" and shutil.which("brew") is not None:
            log_step(
                f"Missing {', '.join(missing)}. Attempting to install with Homebrew..."
            )
            try:
                subprocess.check_call(["brew", "install", "pkg-config", "cmake"])
            except subprocess.CalledProcessError as exc:
                log_step(f"Homebrew installation failed: {exc}")
            else:
                missing = [
                    tool
                    for tool in ("pkg-config", "cmake")
                    if shutil.which(tool) is None
                ]
        if missing:
            names = " and ".join(missing)
            brew = " ".join(missing)
            log_step(
                f"{names} {'are' if len(missing) > 1 else 'is'} required to build native extensions. Install {'them' if len(missing) > 1 else 'it'} (e.g., with Homebrew: 'brew install {brew}') and re-run this script."
            )
            raise SystemExit(1)


