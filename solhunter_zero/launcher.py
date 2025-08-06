#!/usr/bin/env python3
"""Python launcher for SolHunter Zero.

This module replaces the previous ``run.sh`` shell script.  It performs the
same environment preparation steps and launches the appropriate entry point
for the application.
"""
from __future__ import annotations

import atexit
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def _ensure_python() -> None:
    """Ensure the interpreter is at least Python 3.11."""
    if sys.version_info < (3, 11):
        print("Error: Python 3.11 or higher is required.", file=sys.stderr)
        raise SystemExit(1)


def _ensure_config() -> None:
    """Create ``config.toml`` from the example if missing."""
    cfg = ROOT / "config.toml"
    if not cfg.is_file() and (ROOT / "config.example.toml").is_file():
        shutil.copy(ROOT / "config.example.toml", cfg)
        print("Created default config.toml from config.example.toml")


def _detect_gpu() -> bool:
    return (
        subprocess.run(
            [sys.executable, "-m", "solhunter_zero.device", "--check-gpu"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _setup_env() -> None:
    os.environ.setdefault("DEPTH_SERVICE", "true")

    if platform.system() == "Darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if _detect_gpu():
        if platform.system() != "Darwin":
            os.environ.setdefault("GPU_MEMORY_INDEX", "1")
        else:
            os.environ.setdefault("TORCH_DEVICE", "mps")
    print(os.environ.get("TORCH_DEVICE", ""))

    if "RAYON_NUM_THREADS" not in os.environ:
        try:
            if shutil.which("nproc"):
                count = subprocess.check_output(["nproc"], text=True).strip()
            elif shutil.which("getconf"):
                count = subprocess.check_output(
                    ["getconf", "_NPROCESSORS_ONLN"], text=True
                ).strip()
            elif platform.system() == "Darwin":
                count = subprocess.check_output(
                    ["sysctl", "-n", "hw.ncpu"], text=True
                ).strip()
            else:
                count = str(os.cpu_count() or 1)
        except Exception:
            count = "1"
        os.environ["RAYON_NUM_THREADS"] = count


def _check_deps() -> None:
    from scripts import deps

    req, opt = deps.check_deps()
    if not req and not opt:
        return

    missing_opt = opt[:]
    mods = set(missing_opt)
    extras: list[str] = []
    if "orjson" in mods:
        extras.append("fastjson")
    if {"lz4", "zstandard"} & mods:
        extras.append("fastcompress")
    if "msgpack" in mods:
        extras.append("msgpack")
    pkg = "." if not extras else f".[{','.join(extras)}]"

    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    if missing_opt:
        print("Installed optional modules: " + " ".join(missing_opt))

    req_after, opt_after = deps.check_deps()
    if req_after or opt_after:
        if req_after:
            print(
                "Missing required modules: " + " ".join(req_after),
                file=sys.stderr,
            )
        if opt_after:
            print(
                "Missing optional modules: " + " ".join(opt_after),
                file=sys.stderr,
            )
        raise SystemExit(1)


def _check_rpc() -> None:
    url = os.environ.get("SOLANA_RPC_URL")
    if not url:
        return

    import urllib.request

    if url.startswith("ws://"):
        url = "http://" + url[5:]
    elif url.startswith("wss://"):
        url = "https://" + url[6:]

    req = urllib.request.Request(url, method="HEAD")
    for attempt in range(3):
        try:
            urllib.request.urlopen(req, timeout=5)  # nosec B310
            return
        except Exception as exc:  # pragma: no cover - network failure
            if attempt == 2:
                print(
                    f"Error: SOLANA_RPC_URL '{os.environ['SOLANA_RPC_URL']}' is unreachable: {exc}",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            wait = 2**attempt
            print(
                f"Attempt {attempt + 1} failed to reach SOLANA_RPC_URL at {url}: {exc}.",
                f" Retrying in {wait} seconds...",
                file=sys.stderr,
            )
            time.sleep(wait)


def _ensure_cargo() -> None:
    if shutil.which("cargo") is not None:
        return

    print("Installing Rust toolchain via rustup...")
    subprocess.check_call(
        [
            "curl",
            "--proto",
            "=https",
            "--tlsv1.2",
            "-sSf",
            "https://sh.rustup.rs",
            "-o",
            "/tmp/rustup.sh",
        ]
    )
    subprocess.check_call(["sh", "/tmp/rustup.sh", "-y"])
    os.unlink("/tmp/rustup.sh")
    os.environ["PATH"] = (
        f"{Path.home() / '.cargo' / 'bin'}{os.pathsep}{os.environ.get('PATH','')}"
    )
    if shutil.which("cargo") is None:
        print("Error: 'cargo' is not installed.", file=sys.stderr)
        raise SystemExit(1)


def _run_cargo_build(*args: str) -> None:
    try:
        subprocess.check_output(["cargo", "build", *args], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        output = exc.output.decode()
        print(output, file=sys.stderr)
        if "aarch64-apple-darwin" in output:
            print(
                "Hint: run 'rustup target add aarch64-apple-darwin'.", file=sys.stderr
            )
        if "linker" in output and "not found" in output:
            print(
                "Hint: ensure the Xcode command line tools are installed (xcode-select --install).",
                file=sys.stderr,
            )
        raise


def _ensure_route_ffi() -> None:
    uname_s = platform.system()
    uname_m = platform.machine()
    if uname_s == "Darwin":
        libfile = "libroute_ffi.dylib"
    elif (
        uname_s.startswith("MINGW")
        or uname_s.startswith("MSYS")
        or uname_s.startswith("CYGWIN")
        or uname_s == "Windows_NT"
    ):
        libfile = "route_ffi.dll"
    else:
        libfile = "libroute_ffi.so"

    libpath = ROOT / "solhunter_zero" / libfile
    if libpath.exists():
        return

    _run_cargo_build(
        "--manifest-path",
        str(ROOT / "route_ffi" / "Cargo.toml"),
        "--release",
        "--features=parallel",
    )
    src = ROOT / "route_ffi" / "target" / "release" / libfile
    if not src.exists() and uname_s == "Darwin" and uname_m == "arm64":
        src = (
            ROOT / "route_ffi" / "target" / "aarch64-apple-darwin" / "release" / libfile
        )
        if not src.exists():
            print("Rebuilding for aarch64-apple-darwin...")
            _run_cargo_build(
                "--manifest-path",
                str(ROOT / "route_ffi" / "Cargo.toml"),
                "--release",
                "--features=parallel",
                "--target",
                "aarch64-apple-darwin",
            )
            src = (
                ROOT
                / "route_ffi"
                / "target"
                / "aarch64-apple-darwin"
                / "release"
                / libfile
            )
    if src.exists():
        shutil.copy2(src, libpath)
    if not libpath.exists():
        print(f"Error: {libfile} was not copied to solhunter_zero.", file=sys.stderr)
        if uname_s == "Darwin" and uname_m == "arm64":
            print(
                "Try building for macOS arm64 with:\n  cargo build --manifest-path route_ffi/Cargo.toml --release --target aarch64-apple-darwin",
                file=sys.stderr,
            )
        raise SystemExit(1)


def _build_depth_service() -> None:
    _run_cargo_build(
        "--manifest-path",
        str(ROOT / "depth_service" / "Cargo.toml"),
        "--release",
    )


def _launch_metrics() -> subprocess.Popen[bytes] | None:
    log = tempfile.NamedTemporaryFile(delete=False)
    proc = subprocess.Popen(
        [sys.executable, "-m", "solhunter_zero.metrics_aggregator"],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    time.sleep(1)
    if proc.poll() is not None:
        log.seek(0)
        print("metrics_aggregator failed to start", file=sys.stderr)
        print(log.read().decode(), file=sys.stderr)
        log.close()
        os.unlink(log.name)
        raise SystemExit(1)

    def _cleanup() -> None:
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except Exception:
            pass
        os.unlink(log.name)

    atexit.register(_cleanup)
    return proc


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]

    _ensure_python()
    _ensure_config()
    _setup_env()
    _check_deps()
    _check_rpc()
    _ensure_cargo()
    _ensure_route_ffi()
    _build_depth_service()

    no_metrics = False
    args: list[str] = []
    for arg in argv:
        if arg == "--no-metrics":
            no_metrics = True
        else:
            args.append(arg)

    if not no_metrics:
        _launch_metrics()

    first = args[0] if args else ""
    uname_s = platform.system()

    if first == "--daemon":
        subprocess.check_call(
            [sys.executable, "-m", "solhunter_zero.train_cli", "--daemon", *args[1:]]
        )
    elif first == "--start-all" or (not args and uname_s == "Darwin"):
        if first == "--start-all":
            args = args[1:]
        subprocess.check_call(
            [sys.executable, "scripts/start_all.py", "autopilot", *args]
        )
    elif not args or first == "--auto":
        if first == "--auto":
            args = args[1:]
        subprocess.check_call(
            [sys.executable, "-m", "solhunter_zero.main", "--auto", *args]
        )
    else:
        subprocess.check_call([sys.executable, "-m", "solhunter_zero.main", *args])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
