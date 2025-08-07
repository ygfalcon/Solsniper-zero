import platform
from solhunter_zero import bootstrap_utils


def test_ensure_deps_runs_prepare_macos_env(monkeypatch):
    marker = bootstrap_utils.DEPS_MARKER
    marker.unlink(missing_ok=True)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    called = {"prepare": 0}

    def fake_prepare_macos_env(non_interactive=True):
        called["prepare"] += 1
        return {"success": True}

    monkeypatch.setattr(
        "solhunter_zero.macos_setup.prepare_macos_env", fake_prepare_macos_env
    )
    monkeypatch.setattr(
        "solhunter_zero.macos_setup.mac_setup_completed", lambda: False
    )
    monkeypatch.setattr(bootstrap_utils.deps, "check_deps", lambda: ([], []))
    monkeypatch.setattr("solhunter_zero.bootstrap.ensure_route_ffi", lambda: None)
    monkeypatch.setattr("solhunter_zero.bootstrap.ensure_depth_service", lambda: None)

    bootstrap_utils.ensure_deps(ensure_wallet_cli=False)

    assert called["prepare"] == 1


def test_mac_setup_marker_skips_prepare(monkeypatch):
    marker = bootstrap_utils.DEPS_MARKER
    marker.unlink(missing_ok=True)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    called = {"prepare": 0}

    def fake_prepare_macos_env(non_interactive=True):
        called["prepare"] += 1
        return {"success": True}

    monkeypatch.setattr(
        "solhunter_zero.macos_setup.prepare_macos_env", fake_prepare_macos_env
    )
    monkeypatch.setattr(
        "solhunter_zero.macos_setup.mac_setup_completed", lambda: True
    )
    monkeypatch.setattr(bootstrap_utils.deps, "check_deps", lambda: ([], []))
    monkeypatch.setattr("solhunter_zero.bootstrap.ensure_route_ffi", lambda: None)
    monkeypatch.setattr("solhunter_zero.bootstrap.ensure_depth_service", lambda: None)

    bootstrap_utils.ensure_deps(ensure_wallet_cli=False)

    assert called["prepare"] == 0


def test_deps_marker_skips_install(monkeypatch):
    marker = bootstrap_utils.DEPS_MARKER
    marker.unlink(missing_ok=True)

    calls: list[tuple] = []

    monkeypatch.setattr(bootstrap_utils.deps, "check_deps", lambda: (["req"], []))
    monkeypatch.setattr(bootstrap_utils, "_package_missing", lambda pkg: True)
    monkeypatch.setattr(
        bootstrap_utils, "_pip_install", lambda *a, **k: calls.append(a)
    )
    monkeypatch.setattr("solhunter_zero.bootstrap.ensure_route_ffi", lambda: None)
    monkeypatch.setattr("solhunter_zero.bootstrap.ensure_depth_service", lambda: None)
    import types, sys
    dummy_startup = types.SimpleNamespace(check_internet=lambda: [])
    monkeypatch.setitem(sys.modules, "scripts.startup", dummy_startup)
    import importlib
    orig_find_spec = importlib.util.find_spec

    def fake_find_spec(name):
        if name == "req":
            return object()
        return orig_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    bootstrap_utils.ensure_deps(ensure_wallet_cli=False)

    assert marker.exists()
    assert calls

    calls.clear()

    def fail_check():
        raise AssertionError("check_deps should be skipped")

    monkeypatch.setattr(bootstrap_utils.deps, "check_deps", fail_check)
    bootstrap_utils.ensure_deps(ensure_wallet_cli=False)

    assert not calls
    marker.unlink(missing_ok=True)


def test_force_env_var_reinstalls(monkeypatch):
    marker = bootstrap_utils.DEPS_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok")
    calls: list[tuple] = []

    monkeypatch.setattr(bootstrap_utils.deps, "check_deps", lambda: (["req"], []))
    monkeypatch.setattr(bootstrap_utils, "_package_missing", lambda pkg: True)
    monkeypatch.setattr(
        bootstrap_utils, "_pip_install", lambda *a, **k: calls.append(a)
    )
    monkeypatch.setattr("solhunter_zero.bootstrap.ensure_route_ffi", lambda: None)
    monkeypatch.setattr("solhunter_zero.bootstrap.ensure_depth_service", lambda: None)
    monkeypatch.setenv("SOLHUNTER_FORCE_DEPS", "1")
    import types, sys
    dummy_startup = types.SimpleNamespace(check_internet=lambda: [])
    monkeypatch.setitem(sys.modules, "scripts.startup", dummy_startup)
    import importlib
    orig_find_spec = importlib.util.find_spec

    def fake_find_spec(name):
        if name == "req":
            return object()
        return orig_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    bootstrap_utils.ensure_deps(ensure_wallet_cli=False)

    assert calls
    marker.unlink(missing_ok=True)
