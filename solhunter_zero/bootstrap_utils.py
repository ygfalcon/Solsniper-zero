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

    if platform.system() == "Darwin":
        try:
            info = subprocess.check_output(
                [
                    str(python),
                    "-c",
                    (
                        "import json, platform, sys;"
                        "print(json.dumps({'machine': platform.machine(), 'version': sys.version_info[:3]}))"
                    ),
                ],
                text=True,
            )
            data = json.loads(info)
            machine = data.get("machine")
            version = tuple(data.get("version", []))
        except Exception as exc:  # pragma: no cover - hard failure
            print(f"Failed to inspect virtual environment interpreter: {exc}")
            machine = None
            version = (0, 0, 0)

        if machine != "arm64" or version < (3, 11):
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

    if Path(sys.prefix) != venv_dir:
        try:
            os.execv(str(python), [str(python), *sys.argv])
            raise RuntimeError("exec failed")
        except OSError as exc:
            msg = f"Failed to execv {python}: {exc}"
            logging.exception(msg)
            with open(ROOT / "startup.log", "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
            raise


def _pip_install(*args: str, retries: int = 3) -> None:
    """Run ``pip install`` with retries and exponential backoff."""
    errors: list[str] = []
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", *args],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        errors.append(result.stderr.strip() or result.stdout.strip())
        if attempt < retries:
            wait = 2 ** (attempt - 1)
            print(
                f"pip install {' '.join(args)} failed (attempt {attempt}/{retries}). Retrying in {wait} seconds..."
            )
            time.sleep(wait)
    print(f"Failed to install {' '.join(args)} after {retries} attempts:")
    for err in errors:
        if err:
            print(err)
    raise SystemExit(result.returncode)


def ensure_deps(*, install_optional: bool = False) -> None:
    if platform.system() == "Darwin":
        from scripts import mac_setup

        report = mac_setup.ensure_tools()
        if not report.get("success"):
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
