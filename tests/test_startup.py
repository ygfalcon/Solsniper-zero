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

    import importlib  # noqa: E402
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
