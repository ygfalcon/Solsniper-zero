import subprocess
import sys

import pytest


def test_startup_help():
    result = subprocess.run([sys.executable, 'scripts/startup.py', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    out = result.stdout.lower() + result.stderr.lower()
    assert 'usage' in out


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


def test_ensure_keypair_generates_temp(tmp_path, monkeypatch):
    monkeypatch.setenv("KEYPAIR_DIR", str(tmp_path))
    import sys
    sys.modules.pop("solhunter_zero.wallet", None)

    inputs = iter(["", ""])  # path prompt, mnemonic prompt
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    from scripts.startup import ensure_keypair

    ensure_keypair()

    from solhunter_zero import wallet

    assert wallet.list_keypairs() == ["temp"]
    assert wallet.get_active_keypair_name() == "temp"
    assert (tmp_path / "temp.json").exists()


def test_ensure_deps_installs_optional(monkeypatch):
    from scripts import startup

    calls: list[list[str]] = []

    def fake_check_call(cmd):
        calls.append(cmd)
        return 0

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
    monkeypatch.setattr(subprocess, "check_call", fake_check_call)

    startup.ensure_deps()

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


def test_ensure_deps_installs_torch_metal(monkeypatch):
    from scripts import startup

    calls: list[list[str]] = []

    def fake_check_call(cmd):
        calls.append(cmd)
        return 0

    results = [
        ([], ["torch"]),
        ([], []),
    ]

    monkeypatch.setattr(startup.deps, "check_deps", lambda: results.pop(0))
    monkeypatch.setattr(subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(startup.platform, "machine", lambda: "arm64")
    import types, sys
    dummy_torch = types.SimpleNamespace()
    dummy_torch.backends = types.SimpleNamespace()
    dummy_torch.backends.mps = types.SimpleNamespace()
    dummy_torch.backends.mps.is_available = lambda: True
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    startup.ensure_deps()

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

    def fake_check_call(cmd):
        calls.append(cmd)
        return 0

    results = [(["req"], [])]

    monkeypatch.setattr(startup.deps, "check_deps", lambda: results.pop(0))
    monkeypatch.setattr(subprocess, "check_call", fake_check_call)
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
        startup.ensure_deps()

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


def test_main_calls_ensure_endpoints(monkeypatch):
    from scripts import startup

    called: dict[str, object] = {}

    monkeypatch.setattr(startup, "ensure_deps", lambda: None)
    monkeypatch.setattr(startup, "ensure_config", lambda: None)
    monkeypatch.setattr(startup, "ensure_keypair", lambda: None)
    monkeypatch.setattr(startup, "ensure_rpc", lambda: None)
    monkeypatch.setattr(startup, "ensure_cargo", lambda: None)
    monkeypatch.setattr(startup, "ensure_endpoints", lambda cfg: called.setdefault("endpoints", cfg))
    monkeypatch.setattr(startup.os, "execv", lambda *a, **k: (_ for _ in ()).throw(SystemExit(0)))
    import types, sys
    conf = types.SimpleNamespace(
        load_config=lambda: {"dex_base_url": "https://dex.example"},
        validate_config=lambda cfg: cfg,
    )
    monkeypatch.setitem(sys.modules, "solhunter_zero.config", conf)

    with pytest.raises(SystemExit):
        startup.main(["--skip-deps", "--skip-rpc-check"])

    assert "endpoints" in called


def test_main_skips_endpoint_check(monkeypatch):
    from scripts import startup

    called: dict[str, object] = {}

    monkeypatch.setattr(startup, "ensure_deps", lambda: None)
    monkeypatch.setattr(startup, "ensure_config", lambda: None)
    monkeypatch.setattr(startup, "ensure_keypair", lambda: None)
    monkeypatch.setattr(startup, "ensure_rpc", lambda: None)
    monkeypatch.setattr(startup, "ensure_cargo", lambda: None)
    monkeypatch.setattr(startup, "ensure_endpoints", lambda cfg: called.setdefault("endpoints", cfg))
    monkeypatch.setattr(startup.os, "execv", lambda *a, **k: (_ for _ in ()).throw(SystemExit(0)))
    import types, sys
    conf = types.SimpleNamespace(
        load_config=lambda: {"dex_base_url": "https://dex.example"},
        validate_config=lambda cfg: cfg,
    )
    monkeypatch.setitem(sys.modules, "solhunter_zero.config", conf)

    with pytest.raises(SystemExit):
        startup.main(["--skip-deps", "--skip-rpc-check", "--skip-endpoint-check"])

    assert "endpoints" not in called
