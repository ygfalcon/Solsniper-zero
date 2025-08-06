import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_startup_help():
    result = subprocess.run([sys.executable, 'scripts/startup.py', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    out = result.stdout.lower() + result.stderr.lower()
    assert 'usage' in out


def test_mac_startup_prereqs(monkeypatch):
    """Mac-specific startup helpers run without errors."""
    import platform
    import types, sys
    from scripts import startup
    from solhunter_zero import device

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    monkeypatch.delenv("TORCH_DEVICE", raising=False)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    # ensure_venv is a no-op when argv is provided
    startup.ensure_venv([])

    monkeypatch.setattr(startup.deps, "check_deps", lambda: ([], []))
    startup.ensure_deps()

    dummy_torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True)
        ),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(device, "torch", dummy_torch)
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    env = device.ensure_gpu_env()
    assert env.get("TORCH_DEVICE") == "mps"


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
        "elif [ \"$1\" = '-m' ] && [ \"$2\" = 'scripts.threading' ]; then\n"
        "  echo 6\n"
        "else\n"
        "  echo RAYON_NUM_THREADS=$RAYON_NUM_THREADS\n"
        "fi\n"
    )
    os.chmod(bindir / "python3", 0o755)
    os.symlink(bindir / "python3", bindir / "python3.11")

    (bindir / "uname").write_text("#!/bin/bash\necho Darwin\n")
    os.chmod(bindir / "uname", 0o755)

    (bindir / "sysctl").write_text(
        "#!/bin/bash\n"
        "if [ \"$1\" = '-n' ] && [ \"$2\" = 'hw.ncpu' ]; then\n"
        "  echo 6\n"
        "fi\n"
    )
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

    ensure_keypair()

    from solhunter_zero import wallet

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

    ensure_keypair()

    sys.modules.pop("solhunter_zero.wallet", None)
    if hasattr(solhunter_zero, "wallet"):
        delattr(solhunter_zero, "wallet")
    from solhunter_zero import wallet

    from pathlib import Path

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

    def fake_pip_install(*args):
        calls.append([sys.executable, "-m", "pip", "install", *args])

    results = [
        ([], ["torch"]),
        ([], []),
    ]

    monkeypatch.setattr(startup.deps, "check_deps", lambda: results.pop(0))
    monkeypatch.setattr(startup, "_pip_install", fake_pip_install)
    monkeypatch.setattr(subprocess, "check_call", lambda *a, **k: 0)
    monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(startup.platform, "machine", lambda: "arm64")
    import types, sys
    dummy_torch = types.SimpleNamespace()
    dummy_torch.backends = types.SimpleNamespace()
    dummy_torch.backends.mps = types.SimpleNamespace()
    dummy_torch.backends.mps.is_available = lambda: True
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    startup.ensure_deps(install_optional=True)

    assert calls == [
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
    ]
    assert not results


def test_ensure_deps_requires_mps(monkeypatch):
    from scripts import startup

    calls: list[list[str]] = []

    def fake_pip_install(*args):
        calls.append([sys.executable, "-m", "pip", "install", *args])

    results = [(["req"], [])]

    monkeypatch.setattr(startup.deps, "check_deps", lambda: results.pop(0))
    monkeypatch.setattr(startup, "_pip_install", fake_pip_install)
    monkeypatch.setattr(subprocess, "check_call", lambda *a, **k: 0)
    monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(startup.platform, "machine", lambda: "arm64")

    import types, sys, importlib
    dummy_torch = types.SimpleNamespace()
    dummy_torch.backends = types.SimpleNamespace()
    dummy_torch.backends.mps = types.SimpleNamespace()
    dummy_torch.backends.mps.is_available = lambda: False
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)
    monkeypatch.setattr(importlib, "reload", lambda mod: mod)

    with pytest.raises(SystemExit) as excinfo:
        startup.ensure_deps(install_optional=True)

    assert calls[-1] == [
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


def test_ensure_cargo_requires_curl(monkeypatch, capsys):
    from scripts import startup

    def fake_which(cmd):
        return None if cmd in {"cargo", "curl"} else "/usr/bin/" + cmd

    monkeypatch.setattr(startup.shutil, "which", fake_which)
    monkeypatch.setattr(startup.platform, "system", lambda: "Linux")

    with pytest.raises(SystemExit):
        startup.ensure_cargo()

    out = capsys.readouterr().out.lower()
    assert "curl is required" in out


def test_ensure_cargo_requires_brew_on_macos(monkeypatch, capsys):
    from scripts import startup

    def fake_which(cmd):
        return None if cmd in {"cargo", "brew"} else "/usr/bin/" + cmd

    monkeypatch.setattr(startup.shutil, "which", fake_which)
    monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")

    with pytest.raises(SystemExit):
        startup.ensure_cargo()

    out = capsys.readouterr().out.lower()
    assert "homebrew" in out and "mac_setup.py" in out


def test_ensure_cargo_requires_pkg_config_and_cmake(monkeypatch, capsys):
    from scripts import startup

    def fake_which(cmd):
        return None if cmd in {"pkg-config", "cmake"} else "/usr/bin/" + cmd

    monkeypatch.setattr(startup.shutil, "which", fake_which)
    monkeypatch.setattr(startup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(startup.subprocess, "check_call", lambda *a, **k: None)

    with pytest.raises(SystemExit):
        startup.ensure_cargo()

    out = capsys.readouterr().out.lower()
    assert "pkg-config" in out and "cmake" in out


def test_ensure_cargo_installs_pkg_config_and_cmake_with_brew(monkeypatch):
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
        if cmd[:2] == ["brew", "install"]:
            for tool in ("pkg-config", "cmake"):
                installed[tool] = f"/usr/local/bin/{tool}"

    monkeypatch.setattr(startup.shutil, "which", fake_which)
    monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(startup.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(startup.subprocess, "check_call", fake_check_call)

    startup.ensure_cargo()

    assert ["brew", "install", "pkg-config", "cmake"] in calls


def test_main_calls_ensure_endpoints(monkeypatch):
    from scripts import startup

    called: dict[str, object] = {}

    monkeypatch.setattr(startup, "ensure_deps", lambda install_optional=False: None)
    monkeypatch.setattr(startup, "ensure_config", lambda: None)
    monkeypatch.setattr(startup, "ensure_wallet_cli", lambda: None)
    monkeypatch.setattr(startup, "ensure_keypair", lambda: None)
    monkeypatch.setattr(startup, "ensure_default_keypair", lambda: None)
    monkeypatch.setattr(startup, "ensure_rpc", lambda warn_only=False: None)
    monkeypatch.setattr(startup, "ensure_cargo", lambda: None)
    monkeypatch.setattr(startup, "ensure_route_ffi", lambda: None)
    monkeypatch.setattr(startup, "ensure_depth_service", lambda: None)
    monkeypatch.setattr(startup, "ensure_endpoints", lambda cfg: called.setdefault("endpoints", cfg))
    import types, sys
    stub_torch = types.SimpleNamespace(set_default_device=lambda dev: None)
    monkeypatch.setitem(sys.modules, "torch", stub_torch)
    monkeypatch.setattr(startup, "device", types.SimpleNamespace(get_default_device=lambda: "cpu", detect_gpu=lambda: False))
    monkeypatch.setattr(startup.os, "execv", lambda *a, **k: (_ for _ in ()).throw(SystemExit(0)))
    conf = types.SimpleNamespace(
        load_config=lambda path=None: {"dex_base_url": "https://dex.example"},
        validate_config=lambda cfg: cfg,
    )
    monkeypatch.setitem(sys.modules, "solhunter_zero.config", conf)

    with pytest.raises(SystemExit):
        startup.main(["--skip-deps", "--skip-rpc-check", "--skip-preflight"])

    assert "endpoints" in called


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
    )
    monkeypatch.setitem(sys.modules, "solhunter_zero.config", conf)

    with pytest.raises(SystemExit):
        startup.main([
            "--skip-deps",
            "--skip-rpc-check",
            "--skip-endpoint-check",
            "--skip-preflight",
        ])

    assert "endpoints" not in called


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
            "--skip-preflight",
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
        "--skip-preflight",
    ])

    assert ret == 2
    captured = capsys.readouterr()
    assert "out" in captured.out
    assert "err" in captured.err
    assert log_file.exists()
    log_contents = log_file.read_text()
    assert "out" in log_contents
    assert "err" in log_contents


def test_startup_sets_mps_device(monkeypatch):
    monkeypatch.delenv("TORCH_DEVICE", raising=False)

    import platform
    import types, sys, importlib, os

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    dummy_torch = types.SimpleNamespace()
    dummy_torch.backends = types.SimpleNamespace()
    dummy_torch.backends.mps = types.SimpleNamespace()
    dummy_torch.backends.mps.is_available = lambda: True
    dummy_torch.set_default_device = lambda dev: None
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    monkeypatch.setattr("solhunter_zero.device.detect_gpu", lambda: True)
    monkeypatch.setattr("solhunter_zero.device.get_default_device", lambda: "mps")

    import scripts.startup as startup
    startup = importlib.reload(startup)

    monkeypatch.setattr(startup, "ensure_deps", lambda install_optional=False: None)
    monkeypatch.setattr(startup, "ensure_config", lambda: None)
    monkeypatch.setattr(startup, "ensure_wallet_cli", lambda: None)
    monkeypatch.setattr(startup, "ensure_keypair", lambda: None)
    monkeypatch.setattr(startup, "ensure_rpc", lambda warn_only=False: None)
    monkeypatch.setattr(startup, "ensure_cargo", lambda: None)
    monkeypatch.setattr(startup, "ensure_route_ffi", lambda: None)

    monkeypatch.setattr(
        startup.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        startup.os,
        "execv",
        lambda *a, **k: (_ for _ in ()).throw(SystemExit(0)),
    )

    with pytest.raises(SystemExit):
        startup.main(
            [
                "--one-click",
                "--skip-deps",
                "--skip-setup",
                "--skip-endpoint-check",
                "--skip-rpc-check",
                "--skip-preflight",
            ]
        )

    assert os.environ.get("TORCH_DEVICE") == "mps"


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
    )
    monkeypatch.setitem(sys.modules, "solhunter_zero.config", conf)

    def fail_wallet():
        raise SystemExit(5)

    monkeypatch.setattr(startup, "ensure_wallet_cli", fail_wallet)

    ret = startup.main(["--skip-deps", "--skip-rpc-check", "--skip-preflight"])
    assert ret == 5


def test_launch_only_starts_services(monkeypatch):
    import types, sys
    from scripts import startup

    called: dict[str, object] = {}

    def fake_run(cmd, env=None):
        called["cmd"] = cmd
        called["env"] = env
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(startup.subprocess, "run", fake_run)
    monkeypatch.setattr(startup, "ensure_deps", lambda *a, **k: None)
    monkeypatch.setattr(startup, "ensure_cargo", lambda: None)
    monkeypatch.setattr(startup, "ensure_route_ffi", lambda: None)
    monkeypatch.setattr(startup, "ensure_depth_service", lambda: None)
    dummy_device = types.SimpleNamespace(
        ensure_gpu_env=lambda: None,
        detect_gpu=lambda: False,
        get_default_device=lambda: "cpu",
    )
    monkeypatch.setattr(startup, "device", dummy_device)
    dummy_torch = types.SimpleNamespace(set_default_device=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    monkeypatch.setenv("TORCH_DEVICE", "cpu")
    monkeypatch.setenv("EVENT_BUS_URL", "ws://bus")

    code = startup.main(["--launch-only"])
    assert code == 0
    cmd = called.get("cmd")
    assert cmd and cmd[1].endswith("scripts/start_all.py")
    env = called.get("env")
    assert env["TORCH_DEVICE"] == "cpu"
    assert env["EVENT_BUS_URL"] == "ws://bus"
