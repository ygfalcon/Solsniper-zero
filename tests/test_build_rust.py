import pytest

from scripts import build_rust


def test_ensure_route_ffi_triggers_build(monkeypatch, tmp_path):
    monkeypatch.setattr(build_rust, "ROOT", tmp_path)
    monkeypatch.setattr(build_rust.platform, "system", lambda: "Linux")

    calls = {}

    def fake_build(name, cargo_path, output, target=None):
        calls["args"] = (name, cargo_path, output, target)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()

    monkeypatch.setattr(build_rust, "build_rust_component", fake_build)

    build_rust.ensure_route_ffi()

    expected_cargo = tmp_path / "route_ffi" / "Cargo.toml"
    expected_output = tmp_path / "solhunter_zero" / "libroute_ffi.so"

    assert calls["args"] == ("route_ffi", expected_cargo, expected_output, None)


def test_ensure_route_ffi_skips_when_present(monkeypatch, tmp_path):
    monkeypatch.setattr(build_rust, "ROOT", tmp_path)
    monkeypatch.setattr(build_rust.platform, "system", lambda: "Linux")

    libpath = tmp_path / "solhunter_zero" / "libroute_ffi.so"
    libpath.parent.mkdir(parents=True)
    libpath.touch()

    called = False

    def fake_build(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(build_rust, "build_rust_component", fake_build)

    build_rust.ensure_route_ffi()

    assert not called


def test_ensure_depth_service_triggers_build(monkeypatch, tmp_path):
    monkeypatch.setattr(build_rust, "ROOT", tmp_path)
    monkeypatch.setattr(build_rust.platform, "system", lambda: "Linux")

    calls = {}

    def fake_build(name, cargo_path, output, target=None):
        calls["args"] = (name, cargo_path, output, target)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()

    monkeypatch.setattr(build_rust, "build_rust_component", fake_build)

    build_rust.ensure_depth_service()

    expected_cargo = tmp_path / "depth_service" / "Cargo.toml"
    expected_output = tmp_path / "target" / "release" / "depth_service"

    assert calls["args"] == ("depth_service", expected_cargo, expected_output, None)


def test_ensure_depth_service_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(build_rust, "ROOT", tmp_path)
    monkeypatch.setattr(build_rust.platform, "system", lambda: "Linux")

    def fake_build(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(build_rust, "build_rust_component", fake_build)

    with pytest.raises(SystemExit):
        build_rust.ensure_depth_service()


def test_ensure_rust_components_invokes_builders(monkeypatch):
    calls = []

    monkeypatch.setattr(build_rust, "ensure_route_ffi", lambda: calls.append("route"))
    monkeypatch.setattr(
        build_rust, "ensure_depth_service", lambda: calls.append("depth")
    )

    build_rust.ensure_rust_components()

    assert calls == ["route", "depth"]
