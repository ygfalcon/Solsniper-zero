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

from scripts import deps
from . import device
from .device import METAL_EXTRA_INDEX
from .logging_utils import log_startup

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
            log_startup(
                f"pip install {' '.join(args)} succeeded on attempt {attempt}"
            )
            return
        errors.append(result.stderr.strip() or result.stdout.strip())
        if attempt < retries:
            wait = 2 ** (attempt - 1)
            print(
                f"pip install {' '.join(args)} failed (attempt {attempt}/{retries}). Retrying in {wait} seconds..."
            )
            time.sleep(wait)
    msg = f"Failed to install {' '.join(args)} after {retries} attempts:"
    print(msg)
    log_startup(msg)
    for err in errors:
        if err:
            log_startup(err)
            print(err)
    raise SystemExit(result.returncode)


def ensure_deps(*, install_optional: bool = False) -> None:
    if platform.system() == "Darwin":
        from scripts import mac_setup
        report = mac_setup.prepare_macos_env(non_interactive=True)
        mac_setup.apply_brew_env()
        if not report.get("success"):
            for step, info in report.get("steps", {}).items():
                if info.get("status") == "error":
                    fix = mac_setup.MANUAL_FIXES.get(step)
                    if fix:
                        print(f"Manual fix for {step}: {fix}")
            print(
                "macOS environment preparation failed. Please address the issues above and re-run.",
            )
            raise SystemExit(1)
    req, opt = deps.check_deps()
    if req:
        print("Missing required modules: " + ", ".join(req))
    if opt:
        print(
            "Optional modules missing: " + ", ".join(opt) + " (features disabled).",
        )
    if not req and not (install_optional and opt):
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

    need_install = bool(req) or (install_optional and (opt or extra_index))
    if need_install:
        from scripts import startup
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                startup.check_internet()
            except SystemExit as exc:
                raise SystemExit("Unable to establish an internet connection; aborting.") from exc

    if req:
        print("Installing required dependencies...")
        _pip_install(".[uvloop]", *extra_index)
        failed_req = [m for m in req if importlib.util.find_spec(m) is None]
        if failed_req:
            print(
                "Missing required dependencies after installation: "
                + ", ".join(failed_req)
            )
            raise SystemExit(1)

    if install_optional and extra_index:
        try:
            if not device.MPS_SENTINEL.exists():
                device.ensure_torch_with_metal()
        except Exception as exc:
            print(str(exc))
            raise SystemExit(str(exc))
        if "torch" in opt:
            opt.remove("torch")

    if install_optional and opt:
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
        extras: list[str] = []
        if "orjson" in mods:
            extras.append("fastjson")
        if {"lz4", "zstandard"} & mods:
            extras.append("fastcompress")
        if "msgpack" in mods:
            extras.append("msgpack")
        if extras:
            _pip_install(f".[{','.join(extras)}]", *extra_index)
        remaining = mods - {"orjson", "lz4", "zstandard", "msgpack"}
        for name in remaining:
            pkg = mapping.get(name, name.replace("_", "-"))
            _pip_install(pkg, *extra_index)

        missing_opt = [m for m in opt if importlib.util.find_spec(m) is None]
        if missing_opt:
            print(
                "Optional modules missing: "
                + ", ".join(missing_opt)
                + " (features disabled)."
            )


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
        from scripts.mac_setup import apply_brew_env, ensure_tools

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


def build_rust_component(
    name: str, cargo_path: Path, output: Path, *, target: str | None = None
) -> None:
    cmd = ["cargo", "build", "--manifest-path", str(cargo_path), "--release"]
    if target is not None:
        try:
            installed_targets = subprocess.check_output(
                ["rustup", "target", "list", "--installed"], text=True
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("failed to verify rust targets") from exc
        if target not in installed_targets:
            subprocess.check_call(["rustup", "target", "add", target])
        cmd.extend(["--target", target])
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            installed_targets = subprocess.check_output(
                ["rustup", "target", "list", "--installed"], text=True
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("failed to verify rust targets") from exc
        if "aarch64-apple-darwin" not in installed_targets:
            subprocess.check_call(["rustup", "target", "add", "aarch64-apple-darwin"])
        cmd.extend(["--target", "aarch64-apple-darwin"])

    subprocess.check_call(cmd)

    artifact = output.name
    target_dirs = [cargo_path.parent / "target", ROOT / "target"]
    candidates: list[Path] = []
    for base in target_dirs:
        candidates.append(base / "release" / artifact)
        if target is not None:
            candidates.append(base / target / "release" / artifact)
        elif platform.system() == "Darwin" and platform.machine() == "arm64":
            candidates.append(base / "aarch64-apple-darwin" / "release" / artifact)

    built = next((p for p in candidates if p.exists()), None)
    if built is None:
        paths = ", ".join(str(p) for p in candidates)
        raise RuntimeError(f"failed to build {name}: expected {artifact} in {paths}")

    if built.resolve() != output.resolve():
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built, output)

    if not output.exists():
        raise RuntimeError(
            f"{name} build succeeded but {output} is missing. Please build manually."
        )

    if platform.system() == "Darwin":
        try:
            subprocess.check_call(["codesign", "--force", "--sign", "-", str(output)])
        except subprocess.CalledProcessError as exc:
            print(
                f"WARNING: failed to codesign {output}: {exc}. "
                "Please codesign the binary manually if required."
            )


def ensure_route_ffi() -> None:
    libname = "libroute_ffi.dylib" if platform.system() == "Darwin" else "libroute_ffi.so"
    libpath = ROOT / "solhunter_zero" / libname
    if libpath.exists():
        return

    build_rust_component(
        "route_ffi",
        ROOT / "route_ffi" / "Cargo.toml",
        libpath,
    )


def ensure_depth_service() -> None:
    bin_path = ROOT / "target" / "release" / "depth_service"
    if bin_path.exists():
        return

    target = "aarch64-apple-darwin" if platform.system() == "Darwin" else None
    try:
        build_rust_component(
            "depth_service",
            ROOT / "depth_service" / "Cargo.toml",
            bin_path,
            target=target,
        )
    except Exception as exc:  # pragma: no cover - build errors are rare
        hint = ""
        if platform.system() == "Darwin":
            hint = " Hint: run 'scripts/mac_setup.py' to install macOS build tools."
        print(f"Failed to build depth_service: {exc}.{hint}")
        raise SystemExit(1)


def _redirect_stderr(path: Path):
    """Context manager to append stderr output to *path*.

    Subprocesses inherit the redirected file descriptor ensuring that build
    errors are captured in ``startup.log``.
    """
    import contextlib

    @contextlib.contextmanager
    def _cm():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as log:
            old_fd = os.dup(2)
            old_stderr = sys.stderr
            try:
                os.dup2(log.fileno(), 2)
                sys.stderr = log
                yield
            finally:
                sys.stderr = old_stderr
                os.dup2(old_fd, 2)
                os.close(old_fd)

    return _cm()


def ensure_rust_components() -> None:
    """Build required Rust components and capture errors to ``startup.log``."""

    log_path = ROOT / "startup.log"
    for name, func in (
        ("depth_service", ensure_depth_service),
        ("route_ffi", ensure_route_ffi),
    ):
        try:
            with _redirect_stderr(log_path):
                func()
        except SystemExit:
            print(f"Failed to build {name}. See {log_path}")
            raise
        except Exception:
            print(f"Failed to build {name}. See {log_path}")
            raise SystemExit(1)
