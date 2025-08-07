import platform
from solhunter_zero import bootstrap_utils


def test_ensure_deps_runs_prepare_macos_env(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    called = {"prepare": 0}

    def fake_prepare_macos_env(non_interactive=True):
        called["prepare"] += 1
        return {"success": True}

    monkeypatch.setattr(
        "solhunter_zero.macos_setup.prepare_macos_env", fake_prepare_macos_env
    )
    monkeypatch.setattr(bootstrap_utils.deps, "check_deps", lambda: ([], []))

    bootstrap_utils.ensure_deps(ensure_wallet_cli=False)

    assert called["prepare"] == 1
