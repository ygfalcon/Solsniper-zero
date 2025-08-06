import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from solhunter_zero.device import (
    METAL_EXTRA_INDEX,
    TORCH_METAL_VERSION,
    TORCHVISION_METAL_VERSION,
)


def test_startup_help():
    result = subprocess.run([sys.executable, 'scripts/startup.py', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    out = result.stdout.lower() + result.stderr.lower()
    assert 'usage' in out


def test_startup_repair_clears_markers(monkeypatch, capsys):
    import platform
    from scripts import startup

    monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(startup.platform, "machine", lambda: "arm64")

    cargo_marker = startup.ROOT / ".cache" / "cargo-installed"
    cargo_marker.parent.mkdir(parents=True, exist_ok=True)
    cargo_marker.write_text("ok")

    from solhunter_zero import device

    device.MPS_SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    device.MPS_SENTINEL.write_text("ok")

    called = {}

    def fake_prepare(non_interactive=True):
        called["called"] = True
        return {
            "success": False,
            "steps": {"xcode": {"status": "error", "message": "boom"}}
        }

    monkeypatch.setattr("scripts.mac_setup.prepare_macos_env", fake_prepare)
    monkeypatch.setattr("scripts.mac_setup.apply_brew_env", lambda: None)
    monkeypatch.setattr("solhunter_zero.bootstrap.bootstrap", lambda one_click: None)
    monkeypatch.setattr(startup, "ensure_cargo", lambda: None)
    monkeypatch.setattr(startup.device, "ensure_gpu_env", lambda: {})
    monkeypatch.setattr(startup.device, "get_default_device", lambda: "cpu")
    monkeypatch.setattr(startup, "check_internet", lambda: None)
    monkeypatch.setattr(startup, "ensure_rpc", lambda warn_only=False: None)
    monkeypatch.setattr(startup.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0))

    code = startup.main([
        "--repair",
        "--skip-preflight",
        "--skip-rpc-check",
        "--skip-endpoint-check",
        "--skip-setup",
        "--skip-deps",
        "--no-diagnostics",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "Manual fix for xcode" in out
    assert called["called"]
    assert not cargo_marker.exists()
    assert not device.MPS_SENTINEL.exists()


def test_mac_startup_prereqs(monkeypatch):
    """Mac-specific startup helpers run without errors."""
    import platform
    import types, sys
    from scripts import startup
    from solhunter_zero import bootstrap

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    monkeypatch.delenv("TORCH_DEVICE", raising=False)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    # ensure_venv is a no-op when argv is provided
    startup.ensure_venv([])

    monkeypatch.setattr(startup.deps, "check_deps", lambda: ([], []))
    monkeypatch.setattr("scripts.mac_setup.ensure_tools", lambda: {"success": True})
    startup.ensure_deps()

    dummy_torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True)
        ),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(bootstrap.device, "torch", dummy_torch)
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    env = bootstrap.device.ensure_gpu_env()
    assert env.get("TORCH_DEVICE") == "mps"
    assert env.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_start_command_sets_rayon_threads_on_darwin(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    bindir = tmp_path / "bin"
    bindir.mkdir()

    for cmd in ["tee", "awk", "dirname", "tail", "xargs", "ls", "mv", "rm", "date"]:
        src = shutil.which(cmd)
        assert src is not None
        os.symlink(src, bindir / cmd)

    (bindir / "python3").write_text(
        "#!/bin/bash\n"
        "if [ \"$1\" = '-V' ]; then\n"
        "  echo 'Python 3.11.0'\n"
        "elif [ \"$1\" = '-m' ] && [ \"$2\" = 'solhunter_zero.system' ] && [ \"$3\" = 'cpu-count' ]; then\n"
        "  echo 6\n"
        "else\n"
        "  echo RAYON_NUM_THREADS=$RAYON_NUM_THREADS\n"
        "fi\n"
    )
    os.chmod(bindir / "python3", 0o755)
    os.symlink(bindir / "python3", bindir / "python3.11")

    (bindir / "uname").write_text("#!/bin/bash\necho Darwin\n")
    os.chmod(bindir / "uname", 0o755)

    (bindir / "arch").write_text(
        "#!/bin/bash\n"
        "if [ \"$1\" = '-arm64' ]; then\n"
        "  shift\n"
        "fi\n"
        "\"$@\"\n"
    )
    os.chmod(bindir / "arch", 0o755)

    (bindir / "sysctl").write_text("#!/bin/bash\nexit 1\n")
    os.chmod(bindir / "sysctl", 0o755)

    for cmd in ["brew", "rustup"]:
        (bindir / cmd).write_text("#!/bin/bash\nexit 0\n")
        os.chmod(bindir / cmd, 0o755)

    env = {**os.environ, "PATH": str(bindir)}
    env.pop("RAYON_NUM_THREADS", None)

    bash = shutil.which("bash")
    assert bash is not None
    result = subprocess.run(
        [bash, "start.command", "--skip-preflight"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert "RAYON_NUM_THREADS=6" in result.stdout


def test_cluster_setup_assemble(tmp_path):
    cfg = tmp_path / 'cluster.toml'
    cfg.write_text(
        """
event_bus_url = "ws://bus"
broker_url = "redis://host"

[[nodes]]
solana_rpc_url = "url1"
solana_keypair = "kp1"

[[nodes]]
solana_rpc_url = "url2"
solana_keypair = "kp2"
"""
    )

    import importlib
    mod = importlib.import_module('scripts.cluster_setup')
    config = mod.load_cluster_config(str(cfg))
    cmds = mod.assemble_commands(config)

    assert len(cmds) == 2

    cmd1, env1 = cmds[0]
    assert cmd1[1].endswith('start_all.py')
    assert env1['EVENT_BUS_URL'] == 'ws://bus'
    assert env1['BROKER_URL'] == 'redis://host'
    assert env1['SOLANA_RPC_URL'] == 'url1'
    assert env1['SOLANA_KEYPAIR'] == 'kp1'


def test_ensure_keypair_generates_default(tmp_path, monkeypatch):
    monkeypatch.setenv("KEYPAIR_DIR", str(tmp_path))
    import sys
    import solhunter_zero
    sys.modules.pop("solhunter_zero.wallet", None)
    if hasattr(solhunter_zero, "wallet"):
        delattr(solhunter_zero, "wallet")

    from scripts.startup import ensure_keypair

    from solhunter_zero import wallet

    calls = {"count": 0}
    real = wallet.generate_default_keypair

    def wrapped() -> tuple[str, Path]:
        calls["count"] += 1
        return real()

    monkeypatch.setattr(wallet, "generate_default_keypair", wrapped)

    ensure_keypair()

    assert calls["count"] == 1
    assert wallet.list_keypairs() == ["default"]
    assert wallet.get_active_keypair_name() == "default"
    assert (tmp_path / "default.json").exists()
    mn = tmp_path / "default.mnemonic"
    assert mn.exists()
    assert mn.read_text().strip()
    assert (mn.stat().st_mode & 0o777) == 0o600


def test_ensure_keypair_from_json(tmp_path, monkeypatch):
    monkeypatch.setenv("KEYPAIR_DIR", str(tmp_path))
    import sys
    import solhunter_zero
    sys.modules.pop("solhunter_zero.wallet", None)
    if hasattr(solhunter_zero, "wallet"):
        delattr(solhunter_zero, "wallet")

    monkeypatch.delenv("MNEMONIC", raising=False)

    import json
    monkeypatch.setenv("KEYPAIR_JSON", json.dumps([0] * 64))

    from scripts.startup import ensure_keypair

    from solhunter_zero import wallet

    def boom() -> tuple[str, Path]:
        raise AssertionError("should not be called")

    monkeypatch.setattr(wallet, "generate_default_keypair", boom)

    ensure_keypair()

    assert wallet.list_keypairs() == ["default"]
    assert wallet.get_active_keypair_name() == "default"
    assert (Path(wallet.KEYPAIR_DIR) / "default.json").exists()
    # Mnemonic file should not be created when KEYPAIR_JSON provided
    assert not (Path(wallet.KEYPAIR_DIR) / "default.mnemonic").exists()


def test_ensure_deps_installs_optional(monkeypatch):
    from scripts import startup

    calls: list[list[str]] = []

    def fake_pip_install(*args):
        calls.append([sys.executable, "-m", "pip", "install", *args])

    results = [
        (
            [],
            [
                "faiss",
                "sentence_transformers",
                "torch",
                "orjson",
                "lz4",
                "zstandard",
                "msgpack",
            ],
        ),
        ([], []),
    ]
    monkeypatch.setattr(startup.deps, "check_deps", lambda: results.pop(0))
    monkeypatch.setattr(startup, "_pip_install", fake_pip_install)
    monkeypatch.setattr(subprocess, "check_call", lambda *a, **k: 0)

    startup.ensure_deps(install_optional=True)

    assert calls[0] == [
        sys.executable,
        "-m",
        "pip",
        "install",
        ".[fastjson,fastcompress,msgpack]",
    ]
    assert set(c[-1] for c in calls[1:]) == {
        "faiss-cpu",
        "sentence-transformers",
        "torch",
    }
    assert not results  # ensure check_deps called twice


def test_ensure_deps_warns_on_missing_optional(monkeypatch, capsys):
    from scripts import startup

    results = [([], ["orjson", "faiss"])]

    monkeypatch.setattr(startup.deps, "check_deps", lambda: results.pop(0))
    monkeypatch.setattr(startup, "_pip_install", lambda *a, **k: None)
    monkeypatch.setattr(subprocess, "check_call", lambda *a, **k: 0)

    startup.ensure_deps()
    out = capsys.readouterr().out

    assert "Optional modules missing: orjson, faiss (features disabled)." in out
    assert not results


def test_ensure_deps_installs_torch_metal(monkeypatch):
    from scripts import startup

    calls: list[list[str]] = []

    def fake_install():
        calls.append(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch==2.1.0",
                "torchvision==0.16.0",
                "--extra-index-url",
                "https://download.pytorch.org/whl/metal",
            ]
        )

    results = [
        ([], ["torch"]),
        ([], []),
    ]

    monkeypatch.setattr(startup.deps, "check_deps", lambda: results.pop(0))
    monkeypatch.setattr(startup.device, "ensure_torch_with_metal", fake_install)
    monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(startup.platform, "machine", lambda: "arm64")
    monkeypatch.setattr("scripts.mac_setup.ensure_tools", lambda: {"success": True})
    startup.ensure_deps(install_optional=True)

    assert calls == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"torch=={TORCH_METAL_VERSION}",
            f"torchvision=={TORCHVISION_METAL_VERSION}",
            *METAL_EXTRA_INDEX,
        ]
    ]
    assert not results


def test_ensure_deps_requires_mps(monkeypatch):
    from scripts import startup

    calls: list[list[str]] = []

    def fake_pip_install(*args):
        calls.append([sys.executable, "-m", "pip", "install", *args])

    def fake_install():
        calls.append(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "torch==2.1.0",
                "torchvision==0.16.0",
                "--extra-index-url",
                "https://download.pytorch.org/whl/metal",
            ]
        )
        raise RuntimeError("install the Metal wheel manually")

    results = [(["req"], []), ([], [])]

    monkeypatch.setattr(startup.deps, "check_deps", lambda: results.pop(0))
    monkeypatch.setattr(startup, "_pip_install", fake_pip_install)
    monkeypatch.setattr(startup.device, "ensure_torch_with_metal", fake_install)
    monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(startup.platform, "machine", lambda: "arm64")

    monkeypatch.setattr("scripts.mac_setup.ensure_tools", lambda: {"success": True})
    with pytest.raises(SystemExit) as excinfo:
        startup.ensure_deps(install_optional=True)

    assert calls[-1] == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        f"torch=={TORCH_METAL_VERSION}",
        f"torchvision=={TORCHVISION_METAL_VERSION}",
        *METAL_EXTRA_INDEX,
    ]
    assert "install the Metal wheel manually" in str(excinfo.value)


def test_ensure_endpoints_success(monkeypatch):
    from scripts.startup import ensure_endpoints
    import urllib.request

    calls: list[tuple[str, str]] = []

    class Dummy:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=5):
        calls.append((req.full_url, req.get_method()))
        return Dummy()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    cfg = {"dex_base_url": "https://dex.example", "birdeye_api_key": "k"}
    ensure_endpoints(cfg)

    urls = {u for u, _ in calls}
    assert urls == {
        "https://dex.example",
        "https://public-api.birdeye.so/defi/tokenlist",
    }
    assert all(m == "HEAD" for _, m in calls)


def test_ensure_endpoints_failure(monkeypatch, capsys):
    from scripts.startup import ensure_endpoints
    import urllib.request, urllib.error

    def fake_urlopen(req, timeout=5):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    cfg = {"dex_base_url": "https://dex.example"}

    with pytest.raises(SystemExit):
        ensure_endpoints(cfg)

    out = capsys.readouterr().out.lower()
    assert "dex_base_url" in out


def test_ensure_cargo_requires_curl(monkeypatch, capsys, tmp_path):
    from scripts import startup

    def fake_which(cmd):
        return None if cmd in {"cargo", "curl", "brew"} else "/usr/bin/" + cmd

    monkeypatch.setattr(startup.shutil, "which", fake_which)
    monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(startup, "ROOT", tmp_path)

    with pytest.raises(SystemExit):
        startup.ensure_cargo()

    out = capsys.readouterr().out.lower()
    assert "curl is required" in out


def test_ensure_cargo_requires_pkg_config_and_cmake(monkeypatch, capsys, tmp_path):
    from scripts import startup

    def fake_which(cmd):
        return None if cmd in {"pkg-config", "cmake"} else "/usr/bin/" + cmd

    monkeypatch.setattr(startup.shutil, "which", fake_which)
    monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(startup.subprocess, "check_call", lambda *a, **k: None)
    monkeypatch.setattr(startup, "ROOT", tmp_path)

    with pytest.raises(SystemExit):
        startup.ensure_cargo()

    out = capsys.readouterr().out.lower()
    assert "pkg-config" in out and "cmake" in out


def test_ensure_cargo_installs_pkg_config_and_cmake_with_brew(monkeypatch, tmp_path):
    from scripts import startup

    installed = {
        "cargo": "/usr/bin/cargo",
        "pkg-config": None,
        "cmake": None,
        "brew": "/usr/local/bin/brew",
    }

    def fake_which(cmd: str):
        return installed.get(cmd, f"/usr/bin/{cmd}")

    calls: list[list[str]] = []

    def fake_check_call(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["brew", "install"] and "rustup" not in cmd:
            for tool in ("pkg-config", "cmake"):
                installed[tool] = f"/usr/local/bin/{tool}"

    monkeypatch.setattr(startup.shutil, "which", fake_which)
    monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(startup.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(startup.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(startup, "ROOT", tmp_path)

    startup.ensure_cargo()

    assert ["brew", "install", "pkg-config", "cmake"] in calls


def test_ensure_cargo_installs_rustup_with_brew(monkeypatch, tmp_path):
    from scripts import startup

    installed = {"cargo": None, "brew": "/usr/local/bin/brew"}

    def fake_which(cmd: str):
        return installed.get(cmd, f"/usr/bin/{cmd}")

    calls: list[list[str] | str] = []

    def fake_check_call(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["brew", "install"] and "rustup" in cmd:
            installed["cargo"] = "/usr/bin/cargo"
        if cmd == ["cargo", "--version"]:
            return

    monkeypatch.setattr(startup.shutil, "which", fake_which)
    monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(startup.subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(startup, "ROOT", tmp_path)

    startup.ensure_cargo()

    assert ["brew", "install", "rustup"] in calls
    assert ["rustup-init", "-y"] in calls
    assert (tmp_path / ".cache" / "cargo-installed").exists()


def test_ensure_cargo_skips_install_when_cached(monkeypatch, tmp_path, capsys):
    from scripts import startup

    def fake_which(cmd: str):
        return None if cmd == "cargo" else f"/usr/bin/{cmd}"

    marker = tmp_path / ".cache" / "cargo-installed"
    marker.parent.mkdir()
    marker.write_text("ok")

    monkeypatch.setattr(startup.shutil, "which", fake_which)
    monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(startup, "ROOT", tmp_path)

    with pytest.raises(SystemExit):
        startup.ensure_cargo()

    out = capsys.readouterr().out.lower()
    assert "previously installed" in out


def test_main_calls_ensure_endpoints(monkeypatch):
    from scripts import startup

    called: dict[str, object] = {}

    monkeypatch.setattr(startup, "ensure_deps", lambda install_optional=False: None)
    monkeypatch.setattr(startup, "ensure_config", lambda: None)
    monkeypatch.setattr(startup, "ensure_wallet_cli", lambda: None)
    monkeypatch.setattr(startup, "ensure_keypair", lambda: None)
    monkeypatch.setattr(startup, "ensure_rpc", lambda warn_only=False: None)
    monkeypatch.setattr(startup, "ensure_cargo", lambda: None)
    monkeypatch.setattr(startup, "ensure_route_ffi", lambda: None)

    from solhunter_zero import bootstrap as bootstrap_mod
    monkeypatch.setattr(bootstrap_mod, "ensure_route_ffi", lambda: None)
    monkeypatch.setattr(bootstrap_mod, "ensure_depth_service", lambda: None)
    monkeypatch.setattr(bootstrap_mod.device, "torch", dummy_torch)
    monkeypatch.setattr("scripts.mac_setup.ensure_tools", lambda: {"success": True})
    monkeypatch.setattr("scripts.preflight.main", lambda: 0)

    from solhunter_zero import bootstrap as bootstrap_mod
    monkeypatch.setattr(bootstrap_mod, "ensure_route_ffi", lambda: None)
    monkeypatch.setattr(bootstrap_mod, "ensure_depth_service", lambda: None)
    monkeypatch.setattr("scripts.mac_setup.ensure_tools", lambda: {"success": True})
    monkeypatch.setattr("scripts.preflight.main", lambda: 0)
    monkeypatch.setattr(startup, "ensure_depth_service", lambda: None)
    monkeypatch.setattr(startup, "ensure_endpoints", lambda cfg: called.setdefault("endpoints", cfg))
    import types, sys
    stub_torch = types.SimpleNamespace(set_default_device=lambda dev: None)
    monkeypatch.setitem(sys.modules, "torch", stub_torch)
    monkeypatch.setattr(startup, "device", types.SimpleNamespace(get_default_device=lambda: "cpu", detect_gpu=lambda: False))
    monkeypatch.setattr(startup.os, "execv", lambda *a, **k: (_ for _ in ()).throw(SystemExit(0)))
    monkeypatch.setattr(
        startup.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0)
    )
    conf = types.SimpleNamespace(
        load_config=lambda path=None: {"dex_base_url": "https://dex.example"},
        validate_config=lambda cfg: cfg,
        find_config_file=lambda: "config.toml",
    )
    monkeypatch.setitem(sys.modules, "solhunter_zero.config", conf)

    ret = startup.main(["--skip-deps", "--skip-rpc-check", "--skip-preflight"])
    assert "endpoints" in called
    assert ret == 0


def test_main_skips_endpoint_check(monkeypatch):
    from scripts import startup

    called: dict[str, object] = {}

    monkeypatch.setattr(startup, "ensure_deps", lambda install_optional=False: None)
    monkeypatch.setattr(startup, "ensure_config", lambda: None)
    monkeypatch.setattr(startup, "ensure_wallet_cli", lambda: None)
    monkeypatch.setattr(startup, "ensure_keypair", lambda: None)
    monkeypatch.setattr(startup, "ensure_rpc", lambda warn_only=False: None)
    monkeypatch.setattr(startup, "ensure_cargo", lambda: None)
    monkeypatch.setattr(startup, "ensure_endpoints", lambda cfg: called.setdefault("endpoints", cfg))
    import types, sys
    stub_torch = types.SimpleNamespace(set_default_device=lambda dev: None)
    monkeypatch.setitem(sys.modules, "torch", stub_torch)
    monkeypatch.setattr(startup, "device", types.SimpleNamespace(get_default_device=lambda: "cpu", detect_gpu=lambda: False))
    monkeypatch.setattr(startup.os, "execv", lambda *a, **k: (_ for _ in ()).throw(SystemExit(0)))
    conf = types.SimpleNamespace(
        load_config=lambda path=None: {"dex_base_url": "https://dex.example"},
        validate_config=lambda cfg: cfg,
        find_config_file=lambda: "config.toml",
    )
    monkeypatch.setitem(sys.modules, "solhunter_zero.config", conf)

    ret = startup.main([
        "--skip-deps",
        "--skip-rpc-check",
        "--skip-endpoint-check",
        "--skip-preflight",
    ])

    assert "endpoints" not in called
    assert ret == 0


def test_main_preflight_success(monkeypatch):
    from scripts import startup
    import types, sys

    called = {}

    def fake_preflight():
        called["preflight"] = True
        raise SystemExit(0)

    monkeypatch.setattr("scripts.preflight.main", fake_preflight)
    monkeypatch.setattr(startup, "ensure_deps", lambda install_optional=False: None)
    monkeypatch.setattr(startup, "ensure_config", lambda: None)
    monkeypatch.setattr(startup, "ensure_wallet_cli", lambda: None)
    monkeypatch.setattr(startup, "ensure_keypair", lambda: None)
    monkeypatch.setattr(startup, "ensure_rpc", lambda warn_only=False: None)
    monkeypatch.setattr(startup, "ensure_cargo", lambda: None)
    import types as _types, sys
    stub_torch = _types.SimpleNamespace(set_default_device=lambda dev: None)
    monkeypatch.setitem(sys.modules, "torch", stub_torch)
    monkeypatch.setattr(startup, "device", _types.SimpleNamespace(get_default_device=lambda: "cpu", detect_gpu=lambda: False))
    monkeypatch.setattr(startup.os, "execv", lambda *a, **k: (_ for _ in ()).throw(SystemExit(0)))

    with pytest.raises(SystemExit) as exc:
        startup.main([
            "--one-click",
            "--skip-setup",
            "--skip-deps",
        ])

    assert called.get("preflight") is True
    assert exc.value.code == 0


def test_main_preflight_failure(monkeypatch, capsys):
    from scripts import startup
    from pathlib import Path

    def fake_preflight():
        print("out")
        print("err", file=sys.stderr)
        raise SystemExit(2)

    monkeypatch.setattr("scripts.preflight.main", fake_preflight)

    import types
    stub_torch = types.SimpleNamespace(set_default_device=lambda dev: None)
    monkeypatch.setitem(sys.modules, "torch", stub_torch)
    monkeypatch.setattr(startup, "device", types.SimpleNamespace(get_default_device=lambda: "cpu", detect_gpu=lambda: False))
    monkeypatch.setattr(startup, "ensure_cargo", lambda: None)
    monkeypatch.setattr(startup, "ensure_route_ffi", lambda: None)
    monkeypatch.setattr(startup, "ensure_rpc", lambda warn_only=False: None)

    log_file = Path(__file__).resolve().parent.parent / "preflight.log"
    if log_file.exists():
        log_file.unlink()

    ret = startup.main([
        "--one-click",
        "--skip-deps",
        "--skip-setup",
    ])

    assert ret == 2
    captured = capsys.readouterr()
    assert "out" in captured.out
    assert "err" in captured.err
    assert log_file.exists()
    log_contents = log_file.read_text()
    assert "out" in log_contents
    assert "err" in log_contents


def test_preflight_log_rotation(tmp_path):
    from scripts import startup
    log_path = tmp_path / "preflight.log"
    log_path.write_text("x" * 20)
    rotated = tmp_path / "preflight.log.1"

    startup.rotate_preflight_log(log_path, max_bytes=10)

    assert rotated.exists()
    assert not log_path.exists()


def test_startup_sets_mps_device(monkeypatch):
    monkeypatch.delenv("TORCH_DEVICE", raising=False)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    import platform
    import types, sys
    from solhunter_zero import bootstrap

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    dummy_torch = types.SimpleNamespace()
    dummy_torch.backends = types.SimpleNamespace()
    dummy_torch.backends.mps = types.SimpleNamespace()
    dummy_torch.backends.mps.is_available = lambda: True
    dummy_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    dummy_torch.set_default_device = lambda dev: None
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    monkeypatch.setattr(bootstrap, "ensure_venv", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "ensure_deps", lambda install_optional=False: None)
    monkeypatch.setattr(bootstrap, "ensure_keypair", lambda: None)
    monkeypatch.setattr(bootstrap, "ensure_config", lambda: None)
    monkeypatch.setattr(bootstrap, "ensure_route_ffi", lambda: None)
    monkeypatch.setattr(bootstrap, "ensure_depth_service", lambda: None)
    monkeypatch.setattr(bootstrap.device, "torch", dummy_torch)

    bootstrap.bootstrap(one_click=True)

    assert os.environ.get("TORCH_DEVICE") == "mps"
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_wallet_cli_failure_propagates(monkeypatch):
    from scripts import startup

    monkeypatch.setattr(startup, "ensure_deps", lambda: None)
    monkeypatch.setattr(startup, "ensure_config", lambda: None)
    monkeypatch.setattr(startup, "ensure_endpoints", lambda cfg: None)
    monkeypatch.setattr(startup, "ensure_cargo", lambda: None)
    monkeypatch.setattr(startup, "ensure_keypair", lambda: (_ for _ in ()).throw(Exception("should not run")))
    import types, sys
    stub_torch = types.SimpleNamespace(set_default_device=lambda dev: None)
    monkeypatch.setitem(sys.modules, "torch", stub_torch)
    monkeypatch.setattr(startup, "device", types.SimpleNamespace(get_default_device=lambda: "cpu", detect_gpu=lambda: False))
    conf = types.SimpleNamespace(
        load_config=lambda path=None: {"dex_base_url": "https://dex.example"},
        validate_config=lambda cfg: cfg,
        find_config_file=lambda: "config.toml",
    )
    monkeypatch.setitem(sys.modules, "solhunter_zero.config", conf)

    def fail_wallet():
        raise SystemExit(5)

    monkeypatch.setattr(startup, "ensure_wallet_cli", fail_wallet)

    ret = startup.main(["--skip-deps", "--skip-rpc-check", "--skip-preflight"])
    assert ret == 5
