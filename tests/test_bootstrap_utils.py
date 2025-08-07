import platform
from solhunter_zero import bootstrap_utils


def test_ensure_deps_calls_apply_brew_env(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    called = {"prepare": 0, "apply": 0}

    def fake_prepare_macos_env(non_interactive=True):
        called["prepare"] += 1
        return {"success": True}

    def fake_apply_brew_env():
        called["apply"] += 1

    monkeypatch.setattr(
        "solhunter_zero.mac_env.prepare_macos_env", fake_prepare_macos_env
    )
    monkeypatch.setattr(
        "solhunter_zero.mac_env.apply_brew_env", fake_apply_brew_env
    )
    monkeypatch.setattr(bootstrap_utils.deps, "check_deps", lambda: ([], []))

    bootstrap_utils.ensure_deps()

    assert called["prepare"] == 1
    assert called["apply"] == 1
