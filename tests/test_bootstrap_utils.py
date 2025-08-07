from solhunter_zero import bootstrap_utils, platform_utils


def test_ensure_deps_calls_apply_brew_env(monkeypatch):
    monkeypatch.setattr(platform_utils, "is_macos", lambda: True)
    called = {"prepare": 0, "apply": 0}

    def fake_prepare_macos_env(non_interactive=True, force=False):
        called["prepare"] += 1
        return {"success": True}

    def fake_apply_brew_env():
        called["apply"] += 1

    monkeypatch.setattr("scripts.mac_setup.prepare_macos_env", fake_prepare_macos_env)
    monkeypatch.setattr("scripts.mac_setup.apply_brew_env", fake_apply_brew_env)
    monkeypatch.setattr(bootstrap_utils.deps, "check_deps", lambda: ([], []))

    bootstrap_utils.ensure_deps(ensure_wallet_cli=False)

    assert called["prepare"] == 1
    assert called["apply"] == 1
