import subprocess
import sys


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
    monkeypatch.setattr(startup, "check_deps", lambda: results.pop(0))
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

    monkeypatch.setattr(startup, "check_deps", lambda: results.pop(0))
    monkeypatch.setattr(subprocess, "check_call", fake_check_call)
    monkeypatch.setattr(startup.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(startup.platform, "machine", lambda: "arm64")

    startup.ensure_deps()

    assert calls == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "--extra-index-url",
            "https://download.pytorch.org/whl/metal",
        ]
    ]
    assert not results
