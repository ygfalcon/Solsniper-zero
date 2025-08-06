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
os.environ.setdefault("DEPTH_SERVICE", "true")
from solhunter_zero import device

device.ensure_gpu_env()


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
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        missing_tools: list[str] = []
        try:
            if subprocess.run(
                ["xcode-select", "-p"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode != 0:
                missing_tools.append("xcode-select")
        except FileNotFoundError:
            missing_tools.append("xcode-select")
        for cmd in ("brew", "python3.11", "rustup"):
            if shutil.which(cmd) is None:
                missing_tools.append(cmd)
        if missing_tools:
            print(
                "Missing macOS tools: " + ", ".join(missing_tools) + ". Running mac setup..."
            )
            from scripts import mac_setup

            success = mac_setup.prepare_macos_env(non_interactive=True)
            mac_setup.apply_brew_env()
            if not success:
                print(
                    "macOS environment preparation failed; continuing without required tools",
                    file=sys.stderr,
                )

    req, opt = deps.check_deps()
    if not req and not install_optional:
        if opt:
            print(
                "Optional modules missing: "
                + ", ".join(opt)
                + " (features disabled)."
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
        extra_index = [
            "--extra-index-url",
            "https://download.pytorch.org/whl/metal",
        ]

    if req:
        print("Installing required dependencies...")
        _pip_install(".[uvloop]", *extra_index)

    if install_optional and "torch" in opt and extra_index:
        print(
            "Installing torch==2.1.0 and torchvision==0.16.0 for macOS arm64 with Metal support..."
        )
        _pip_install("torch==2.1.0", "torchvision==0.16.0", *extra_index)
        opt.remove("torch")

    if install_optional and extra_index:
        import importlib
        import torch

        if not torch.backends.mps.is_available():
            print(
                "MPS backend not available; attempting to reinstall Metal wheel..."
            )
            try:
                _pip_install(
                    "--force-reinstall",
                    "torch==2.1.0",
                    "torchvision==0.16.0",
                    *extra_index,
                )
            except SystemExit:
                print("Failed to reinstall torch with Metal wheels.")
            importlib.reload(torch)
            if not torch.backends.mps.is_available():
                raise SystemExit(
                    "MPS backend still not available. Please install the Metal wheel manually:\n"
                    "pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url "
                    "https://download.pytorch.org/whl/metal"
                )

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
            "Missing required dependencies after installation: "
            + ", ".join(req_after)
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
    cfg_file = ROOT / "config.toml"
    if not cfg_file.exists():
        template = ROOT / "config" / "default.toml"
        if template.exists():
            shutil.copy(template, cfg_file)

    import tomllib
    try:
        import tomli_w  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tomli-w"])
            import tomli_w  # type: ignore
        except Exception as exc:  # pragma: no cover - installation failure
            print(f"Failed to install 'tomli-w': {exc}")
            raise SystemExit(1)

    from solhunter_zero.config import apply_env_overrides, validate_config

    if cfg_file.exists():
        with cfg_file.open("rb") as fh:
            cfg = tomllib.load(fh)
    else:
        cfg = {}

    cfg = apply_env_overrides(cfg)
    try:
        cfg = validate_config(cfg)
    except ValueError as exc:  # pragma: no cover - config validation
        print(f"Invalid configuration: {exc}")
        raise SystemExit(1)

    with cfg_file.open("wb") as fh:
        fh.write(tomli_w.dumps(cfg).encode("utf-8"))


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
        print("Failed to install 'solhunter-wallet'. Please run 'pip install .' manually.")
        raise

    if shutil.which("solhunter-wallet") is None:
        print("'solhunter-wallet' still not available after installation. Aborting.")
        raise SystemExit(1)


def ensure_default_keypair() -> None:
    """Create and select a default keypair if none is active.

    The helper checks for ``keypairs/active`` and, when missing, invokes the
    ``scripts/setup_default_keypair.sh`` helper via ``subprocess``.  The shell
    script prints the mnemonic only when it generates one.  In that case the
    mnemonic is echoed here with guidance to store it securely.
    """

    active_file = ROOT / "keypairs" / "active"
    if active_file.exists():
        return

    setup_script = ROOT / "scripts" / "setup_default_keypair.sh"
    try:
        result = subprocess.run(
            ["bash", str(setup_script)], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Failed to run {setup_script}: {exc}")
        raise SystemExit(1)

    mnemonic = result.stdout.strip()
    if mnemonic:
        print(f"Generated mnemonic: {mnemonic}")
        print("Please store this mnemonic securely; it will not be shown again.")


def ensure_keypair() -> None:
    """Ensure a usable keypair exists and is selected.

    When no keypairs are present a new one is derived.  The generated mnemonic
    is written to ``keypairs/default.mnemonic`` with mode ``600`` so it is only
    readable by the current user.  In ``--one-click`` mode console output is
    suppressed and the storage path is logged instead.
    """

    import json
    import logging

    from solhunter_zero import wallet

    log = logging.getLogger(__name__)
    one_click = os.getenv("AUTO_SELECT_KEYPAIR") == "1"

    def _msg(msg: str) -> None:
        if one_click:
            log.info(msg)
        else:
            print(msg)

    def _wallet_cmd(*args: str) -> list[str]:
        if shutil.which("solhunter-wallet") is not None:
            return ["solhunter-wallet", *args]
        return [sys.executable, "-m", "solhunter_zero.wallet_cli", *args]

    keypairs = wallet.list_keypairs()
    active = wallet.get_active_keypair_name()
    if keypairs:
        if len(keypairs) == 1 and active is None:
            name = keypairs[0]
            wallet.select_keypair(name)
            _msg(f"Automatically selected keypair '{name}'.")
        return

    keypair_json = os.environ.get("KEYPAIR_JSON")

    # Skip mnemonic generation entirely when KEYPAIR_JSON is provided
    if keypair_json:
        try:
            data = json.loads(keypair_json)
            if isinstance(data, list):
                wallet.save_keypair("default", data)
            else:
                raise ValueError
        except Exception:
            subprocess.check_call(_wallet_cmd("save", "default", keypair_json))
        wallet.select_keypair("default")
        _msg("Keypair saved from KEYPAIR_JSON and selected as 'default'.")
        return

    mnemonic, mnemonic_path = wallet.generate_default_keypair()

    _msg("Generated mnemonic and keypair 'default'.")
    _msg(f"Mnemonic stored at {mnemonic_path}.")
    if not one_click:
        _msg("Please store this mnemonic securely; it will not be shown again.")
    return


def ensure_endpoints(cfg: dict) -> None:
    """Ensure HTTP endpoints in ``cfg`` are reachable.

    The configuration may specify several service URLs such as
    ``DEX_BASE_URL`` or custom metrics endpoints.  This function attempts a
    ``HEAD`` request to each HTTP(S) URL and aborts startup if any service is
    unreachable.  BirdEye is only checked when an API key is configured.
    """

    import urllib.error
    import urllib.request
    import time

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
        req = urllib.request.Request(url, method="HEAD")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=5):  # nosec B310
                    break
            except urllib.error.URLError as exc:  # pragma: no cover - network failure
                if attempt == 2:
                    print(
                        f"Failed to reach {name} at {url} after 3 attempts: {exc}."
                        " Check your network connection or configuration."
                    )
                    failed.append(name)
                else:
                    wait = 2**attempt
                    print(
                        f"Attempt {attempt + 1} failed for {name} at {url}: {exc}."
                        f" Retrying in {wait} seconds..."
                    )
                    time.sleep(wait)

    if failed:
        raise SystemExit(1)


def ensure_rpc(*, warn_only: bool = False) -> None:
    """Send a simple JSON-RPC request to ensure the Solana RPC is reachable."""
    rpc_url = os.environ.get(
        "SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"
    )
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
    if shutil.which("cargo") is None:
        if shutil.which("curl") is None:
            print(
                "curl is required to install the Rust toolchain. "
                "Install it (e.g., with Homebrew: 'brew install curl') and re-run this script.",
            )
            raise SystemExit(1)
        if platform.system() == "Darwin":
            if shutil.which("brew") is None:
                print(
                    "Homebrew is required to install the Rust toolchain. "
                    "Install it by running scripts/mac_setup.py and re-run this script.",
                )
                raise SystemExit(1)
            try:
                subprocess.check_call(
                    ["xcode-select", "-p"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                print(
                    "Xcode command line tools are required. Launching installer..."
                )
                try:
                    subprocess.check_call(["xcode-select", "--install"])
                except subprocess.CalledProcessError as exc:  # pragma: no cover - hard failure
                    print(
                        f"Failed to start Xcode command line tools installer: {exc}"
                    )
                else:
                    print(
                        "The installer may prompt for confirmation; "
                        "after it finishes, re-run this script to resume."
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
    if (
        installed
        and platform.system() == "Darwin"
        and platform.machine() == "arm64"
    ):
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
                    tool for tool in ("pkg-config", "cmake") if shutil.which(tool) is None
                ]
        if missing:
            names = " and ".join(missing)
            brew = " ".join(missing)
            print(
                f"{names} {'are' if len(missing) > 1 else 'is'} required to build native extensions. "
                f"Install {'them' if len(missing) > 1 else 'it'} (e.g., with Homebrew: 'brew install {brew}') and re-run this script."
            )
            raise SystemExit(1)


def build_rust_component(name: str, cargo_path: Path, output: Path) -> None:
    """Build a Rust component and ensure its artifact exists.

    On Apple Silicon targets the function verifies the ``aarch64-apple-darwin``
    target is installed and builds for it explicitly. The compiled binary or
    library is copied to ``output`` when necessary and automatically codesigned
    on macOS. A ``RuntimeError`` is raised when the expected artifact cannot be
    located after the build completes.
    """

    cmd = ["cargo", "build", "--manifest-path", str(cargo_path), "--release"]
    if platform.system() == "Darwin" and platform.machine() == "arm64":
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
        if platform.system() == "Darwin" and platform.machine() == "arm64":
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
        subprocess.check_call(["codesign", "--force", "--sign", "-", str(output)])


def ensure_route_ffi() -> None:
    """Ensure the ``route_ffi`` Rust library is built and copied locally."""

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
    """Build the ``depth_service`` binary if missing."""

    bin_path = ROOT / "target" / "release" / "depth_service"
    if bin_path.exists():
        return

    build_rust_component(
        "depth_service",
        ROOT / "depth_service" / "Cargo.toml",
        bin_path,
    )


def main(argv: list[str] | None = None) -> int:
    if argv is not None:
        os.environ["SOLHUNTER_SKIP_VENV"] = "1"

    parser = argparse.ArgumentParser(description="Guided setup and launch")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency check")
    parser.add_argument(
        "--full-deps",
        action="store_true",
        help="Install optional dependencies",
    )
    parser.add_argument("--skip-setup", action="store_true", help="Skip config and wallet prompts")
    parser.add_argument(
        "--skip-rpc-check", action="store_true", help="Skip Solana RPC availability check"
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

    device.ensure_gpu_env()
    import torch

    torch.set_default_device(device.get_default_device())
    gpu_available = device.detect_gpu()
    gpu_device = str(device.get_default_device()) if gpu_available else "none"
    os.environ["SOLHUNTER_GPU_AVAILABLE"] = "1" if gpu_available else "0"
    os.environ["SOLHUNTER_GPU_DEVICE"] = gpu_device
    config_path: str | None = None
    active_keypair: str | None = None
    rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")

    if not args.skip_setup:
        from solhunter_zero.config import load_config, validate_config

        config_path = os.getenv("SOLHUNTER_CONFIG")
        if not config_path:
            for name in ("config.yaml", "config.yml", "config.toml"):
                if Path(name).is_file():
                    config_path = name
                    break
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
        from scripts import preflight

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
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
    args_list = argv or sys.argv[1:]
    try:
        code = main(argv)
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
