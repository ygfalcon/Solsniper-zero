import platform
from solhunter_zero import bootstrap_utils


def test_ensure_deps_prepares_macos_env_once(monkeypatch, tmp_path):
    marker = bootstrap_utils.DEPS_MARKER
    marker.unlink(missing_ok=True)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    mac_marker = tmp_path / ".cache" / "mac_setup_complete"
    monkeypatch.setattr(
        "solhunter_zero.macos_setup.MAC_SETUP_MARKER", mac_marker
    )

    called: dict[str, list | int] = {"prepare": 0, "tools": []}
    report = {"success": True, "steps": {}}

    def fake_prepare_macos_env(non_interactive=True):
        called["prepare"] += 1
        mac_marker.parent.mkdir(parents=True)
        mac_marker.write_text("ok")
        return report

    def fake_ensure_tools(*, non_interactive=True, setup_report=None):
        called["tools"].append(setup_report)

    monkeypatch.setattr(
        "solhunter_zero.macos_setup.prepare_macos_env", fake_prepare_macos_env
    )
    monkeypatch.setattr(
        "solhunter_zero.macos_setup.ensure_tools", fake_ensure_tools
    )
    monkeypatch.setattr(bootstrap_utils.deps, "check_deps", lambda: ([], []))
    monkeypatch.setattr("solhunter_zero.bootstrap.ensure_route_ffi", lambda: None)
    monkeypatch.setattr("solhunter_zero.bootstrap.ensure_depth_service", lambda: None)

    bootstrap_utils.ensure_deps(ensure_wallet_cli=False)

    assert called["prepare"] == 1
    assert called["tools"] == [report]

    bootstrap_utils.ensure_deps(ensure_wallet_cli=False)

    assert called["prepare"] == 1
    assert called["tools"] == [report, None]


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
    dummy_startup = types.SimpleNamespace(check_internet=lambda: None)
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
    dummy_startup = types.SimpleNamespace(check_internet=lambda: None)
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
