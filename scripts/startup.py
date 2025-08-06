#!/usr/bin/env python3
"""Interactive startup script for SolHunter Zero."""

from __future__ import annotations

import sys

import argparse
import os
import platform
import subprocess
import shutil
import time
import contextlib
import io
from pathlib import Path
import json

from scripts import deps

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from solhunter_zero import env  # noqa: E402

env.load_env_file(ROOT / ".env")
os.environ.setdefault("DEPTH_SERVICE", "true")
from solhunter_zero import device  # noqa: E402
from solhunter_zero.device import METAL_EXTRA_INDEX  # noqa: E402

if platform.system() == "Darwin" and platform.machine() == "x86_64":
    script = Path(__file__).resolve()
    cmd = ["arch", "-arm64", sys.executable, str(script), *sys.argv[1:]]
    try:
        os.execvp(cmd[0], cmd)
    except OSError as exc:  # pragma: no cover - hard failure
        msg = (
            f"Failed to re-exec {script.name} via 'arch -arm64': {exc}\n"
            "Please use the unified entry point (scripts/launcher.py or start.command)."
        )
        raise SystemExit(msg)

MAX_PREFLIGHT_LOG_SIZE = 1_000_000  # 1 MB


def rotate_preflight_log(
    path: Path | None = None, max_bytes: int = MAX_PREFLIGHT_LOG_SIZE
) -> None:
    """Rotate or truncate the preflight log before writing new output.

    When ``path`` exists and exceeds ``max_bytes`` it is moved to ``.1``.
    Otherwise the file is truncated to start fresh for the current run.
    """

    path = path or ROOT / "preflight.log"
    if not path.exists():
        return
    try:
        if path.stat().st_size > max_bytes:
            backup = path.with_suffix(path.suffix + ".1")
            path.replace(backup)
        else:
            path.write_text("")
    except OSError:
        pass


def ensure_venv(argv: list[str] | None) -> None:
    """Create a local virtual environment and re-invoke the script inside it.

    The environment is created in ``ROOT/.venv`` using ``python -m venv``.  If
    the script is not already executing from that interpreter, ``execv`` is used
    to restart the current process with ``.venv/bin/python`` (or the Windows
    equivalent).  When ``argv`` is not ``None`` the function assumes it is being
    called from tests and does nothing to avoid spawning subprocesses.  If the
    environment exists but the interpreter is missing or not executable the
    directory is removed and recreated before continuing.
    """

    if argv is not None:  # avoid side effects during tests
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
        os.execv(str(python), [str(python), *sys.argv])


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
    if not req and not install_optional:
        if opt:
            print(
                "Optional modules missing: " + ", ".join(opt) + " (features disabled)."
            )
        return

    # Ensure ``pip`` is available before attempting installations.
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

    if req:
        print("Installing required dependencies...")
        _pip_install(".[uvloop]", *extra_index)

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

    req_after, opt_after = deps.check_deps()
    missing_opt = list(opt_after)
    if req_after:
        print(
            "Missing required dependencies after installation: " + ", ".join(req_after)
        )
    if missing_opt:
        print(
            "Optional modules missing: "
            + ", ".join(missing_opt)
            + " (features disabled)."
        )
    if req_after:
        raise SystemExit(1)


def ensure_config() -> None:
    """Ensure a configuration file exists and is valid."""
    from solhunter_zero.config_bootstrap import ensure_config as _ensure_config

    _ensure_config()


def ensure_wallet_cli() -> None:
    """Ensure the ``solhunter-wallet`` CLI is available.

    If the command is missing, attempt to install the current package which
    provides it. On failure, instruct the user and abort gracefully.
    """
    if shutil.which("solhunter-wallet") is not None:
        return

    print("'solhunter-wallet' command not found. Installing the package...")
    try:
        _pip_install(".")
    except SystemExit:
        print(
            "Failed to install 'solhunter-wallet'. Please run 'pip install .' manually."
        )
        raise

    if shutil.which("solhunter-wallet") is None:
        print("'solhunter-wallet' still not available after installation. Aborting.")
        raise SystemExit(1)

def ensure_keypair() -> None:
    """Ensure a usable keypair exists and is selected."""

    import logging
    from pathlib import Path

    from solhunter_zero import wallet

    log = logging.getLogger(__name__)
    one_click = os.getenv("AUTO_SELECT_KEYPAIR") == "1"

    def _msg(msg: str) -> None:
        if one_click:
            log.info(msg)
        else:
            print(msg)

    keypair_json = os.environ.get("KEYPAIR_JSON")
    result = wallet.setup_default_keypair()
    name, mnemonic_path = result.name, result.mnemonic_path
    keypair_path = Path(wallet.KEYPAIR_DIR) / f"{name}.json"

    if keypair_json:
        _msg("Keypair saved from KEYPAIR_JSON and selected as 'default'.")
        _msg(f"Keypair stored at {keypair_path}.")
    elif mnemonic_path:
        _msg(f"Generated mnemonic and keypair '{name}'.")
        _msg(f"Keypair stored at {keypair_path}.")
        _msg(f"Mnemonic stored at {mnemonic_path}.")
        if not one_click:
            _msg("Please store this mnemonic securely; it will not be shown again.")
    else:
        _msg(f"Using keypair '{name}'.")

    return


def ensure_endpoints(cfg: dict) -> None:
    """Ensure HTTP endpoints in ``cfg`` are reachable.

    The configuration may specify several service URLs such as
    ``DEX_BASE_URL`` or custom metrics endpoints.  This function attempts a
    ``HEAD`` request to each HTTP(S) URL and aborts startup if any service is
    unreachable.  BirdEye is only checked when an API key is configured.
    """

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

    failed: list[str] = []
    for name, url in urls.items():
        try:
            check_endpoint(url)
        except urllib.error.URLError as exc:  # pragma: no cover - network failure
            print(
                f"Failed to reach {name} at {url} after 3 attempts: {exc}."
                " Check your network connection or configuration."
            )
            failed.append(name)

    if failed:
        raise SystemExit(1)


def ensure_rpc(*, warn_only: bool = False) -> None:
    """Send a simple JSON-RPC request to ensure the Solana RPC is reachable."""
    rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    if not os.environ.get("SOLANA_RPC_URL"):
        print(f"Using default RPC URL {rpc_url}")

    import json
    import urllib.request
    import time

    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "getHealth"}).encode()
    req = urllib.request.Request(
        rpc_url, data=payload, headers={"Content-Type": "application/json"}
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310
                resp.read()
                break
        except Exception as exc:  # pragma: no cover - network failure
            if attempt == 2:
                msg = (
                    f"Failed to contact Solana RPC at {rpc_url} after 3 attempts: {exc}."
                    " Please ensure the endpoint is reachable or set SOLANA_RPC_URL to a valid RPC."
                )
                if warn_only:
                    print(f"Warning: {msg}")
                    return
                print(msg)
                raise SystemExit(1)
            wait = 2**attempt
            print(
                f"Attempt {attempt + 1} failed to contact Solana RPC at {rpc_url}: {exc}.",
                f" Retrying in {wait} seconds...",
            )
            time.sleep(wait)


def ensure_cargo() -> None:
    installed = False
    if platform.system() == "Darwin":
        from scripts import mac_setup

        mac_setup.ensure_tools()
    if shutil.which("cargo") is None:
        if shutil.which("curl") is None:
            print(
                "curl is required to install the Rust toolchain. "
                "Install it (e.g., with Homebrew: 'brew install curl') and re-run this script.",
            )
            raise SystemExit(1)
        print("Installing Rust toolchain via rustup...")
        subprocess.check_call(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
            shell=True,
        )
        installed = True
    cargo_bin = Path.home() / ".cargo" / "bin"
    os.environ["PATH"] = f"{cargo_bin}{os.pathsep}{os.environ.get('PATH', '')}"
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
                f"Missing {', '.join(missing)}. Attempting to install with Homebrew..."
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
                f"Install {'them' if len(missing) > 1 else 'it'} (e.g., with Homebrew: 'brew install {brew}') and re-run this script."
            )
            raise SystemExit(1)


def build_rust_component(
    name: str, cargo_path: Path, output: Path, *, target: str | None = None
) -> None:
    """Build a Rust component and ensure its artifact exists.

    When ``target`` is provided the required Rust target is ensured and used for
    the build. The compiled binary or library is copied to ``output`` when
    necessary and automatically codesigned on macOS. A ``RuntimeError`` is
    raised when the expected artifact cannot be located after the build
    completes.
    """

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
    """Ensure the ``route_ffi`` Rust library is built and copied locally."""

    libname = (
        "libroute_ffi.dylib" if platform.system() == "Darwin" else "libroute_ffi.so"
    )
    libpath = ROOT / "solhunter_zero" / libname
    if libpath.exists():
        return

    build_rust_component(
        "route_ffi",
        ROOT / "route_ffi" / "Cargo.toml",
        libpath,
    )


def ensure_depth_service() -> None:
    """Build the ``depth_service`` binary if missing."""

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


def main(argv: list[str] | None = None) -> int:
    if argv is not None:
        os.environ["SOLHUNTER_SKIP_VENV"] = "1"
    parser = argparse.ArgumentParser(description="Guided setup and launch")
    parser.add_argument(
        "--skip-deps", action="store_true", help="Skip dependency check"
    )
    parser.add_argument(
        "--full-deps",
        action="store_true",
        help="Install optional dependencies",
    )
    parser.add_argument(
        "--skip-setup", action="store_true", help="Skip config and wallet prompts"
    )
    parser.add_argument(
        "--skip-rpc-check",
        action="store_true",
        help="Skip Solana RPC availability check",
    )
    parser.add_argument(
        "--skip-endpoint-check",
        action="store_true",
        help="Skip HTTP endpoint availability check",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip environment preflight checks",
    )
    parser.add_argument(
        "--one-click",
        action="store_true",
        help="Enable fully automated non-interactive startup",
    )
    parser.add_argument(
        "--allow-rosetta",
        action="store_true",
        help="Allow running under Rosetta (no Metal acceleration)",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print system diagnostics and exit",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Suppress post-run diagnostics collection",
    )
    args, rest = parser.parse_known_args(argv)

    if args.diagnostics:
        from scripts import diagnostics

        diagnostics.main()
        return 0
    if args.one_click:
        rest = ["--non-interactive", *rest]

    if args.skip_deps:
        os.environ["SOLHUNTER_SKIP_DEPS"] = "1"
    if args.full_deps:
        os.environ["SOLHUNTER_INSTALL_OPTIONAL"] = "1"
    if args.skip_setup:
        os.environ["SOLHUNTER_SKIP_SETUP"] = "1"

    if sys.version_info < (3, 11):
        print(
            "Python 3.11 or higher is required. "
            "Please install Python 3.11 following the instructions in README.md."
        )
        return 1

    if platform.system() == "Darwin" and platform.machine() == "x86_64":
        print("Warning: running under Rosetta; Metal acceleration unavailable.")
        if not args.allow_rosetta:
            print("Use '--allow-rosetta' to continue anyway.")
            return 1

    from solhunter_zero.bootstrap import bootstrap

    bootstrap(one_click=args.one_click)

    from solhunter_zero import device
    import torch

    torch.set_default_device(device.get_default_device())
    gpu_available = device.detect_gpu()
    gpu_device = str(device.get_default_device()) if gpu_available else "none"
    if gpu_device == "none":
        print(
            "No GPU backend detected. Install a Metal-enabled PyTorch build or run "
            "scripts/mac_setup.py to enable GPU support."
        )
    os.environ["SOLHUNTER_GPU_AVAILABLE"] = "1" if gpu_available else "0"
    os.environ["SOLHUNTER_GPU_DEVICE"] = gpu_device
    config_path: str | None = None
    active_keypair: str | None = None
    rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")

    if not args.skip_setup:
        from solhunter_zero.config import (
            load_config,
            validate_config,
            find_config_file,
        )

        config_path = find_config_file()
        cfg = load_config(config_path)
        cfg = validate_config(cfg)
        if not args.skip_endpoint_check:
            ensure_endpoints(cfg)
        try:
            ensure_wallet_cli()
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 1
        from solhunter_zero import wallet

        active_keypair = wallet.get_active_keypair_name()

    if not args.skip_rpc_check:
        ensure_rpc(warn_only=args.one_click)
        rpc_status = "reachable"
    else:
        rpc_status = "skipped"

    ensure_cargo()

    run_preflight = args.one_click or not args.skip_preflight
    if run_preflight:
        rotate_preflight_log()
        from scripts import preflight

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with (
            contextlib.redirect_stdout(stdout_buf),
            contextlib.redirect_stderr(stderr_buf),
        ):
            try:
                preflight.main()
            except SystemExit as exc:  # propagate non-zero exit codes
                code = exc.code if isinstance(exc.code, int) else 1
            else:
                code = 0
        out = stdout_buf.getvalue()
        err = stderr_buf.getvalue()
        sys.stdout.write(out)
        sys.stderr.write(err)
        try:
            with open(ROOT / "preflight.log", "a", encoding="utf-8") as log:
                log.write(out)
                log.write(err)
        except OSError:
            pass
        if code:
            return code
    print("Startup summary:")
    print(f"  Config file: {config_path or 'none'}")
    print(f"  Active keypair: {active_keypair or 'none'}")
    print(f"  GPU device: {gpu_device}")
    print(f"  RPC endpoint: {rpc_url} ({rpc_status})")

    proc = subprocess.run(
        [sys.executable, "-m", "solhunter_zero.main", "--auto", *rest]
    )

    if not args.no_diagnostics:
        from scripts import diagnostics

        info = diagnostics.collect()
        summary = ", ".join(f"{k}={v}" for k, v in info.items())
        print(f"Diagnostics summary: {summary}")
        out_path = Path("diagnostics.json")
        try:
            out_path.write_text(json.dumps(info, indent=2))
        except Exception:
            pass
        else:
            print(f"Full diagnostics written to {out_path}")

    return proc.returncode


def run(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if "--one-click" not in args_list:
        args_list.insert(0, "--one-click")
    try:
        code = main(args_list)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    except Exception:
        if "--no-diagnostics" not in args_list:
            from scripts import diagnostics

            diagnostics.main()
        raise
    if code and "--no-diagnostics" not in args_list:
        from scripts import diagnostics

        diagnostics.main()
    return code or 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
