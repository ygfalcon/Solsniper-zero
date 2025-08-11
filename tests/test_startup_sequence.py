import logging
import pytest


def _setup_fake_log_startup(monkeypatch):
    import solhunter_zero.logging_utils as lu

    def fake_log_startup(msg, path=lu.STARTUP_LOG):
        logging.getLogger("startup").info(msg)

    monkeypatch.setattr(lu, "log_startup", fake_log_startup)
    return fake_log_startup


@pytest.mark.usefixtures("tmp_path")
def test_startup_log_sequence(monkeypatch, caplog, tmp_path):
    import solhunter_zero.launcher as launcher
    import types, sys

    caplog.set_level(logging.INFO, logger="startup")

    fake_log_startup = _setup_fake_log_startup(monkeypatch)

    # Bypass platform/arg handling and heavy operations
    monkeypatch.setattr(launcher, "_ensure_arm64_python", lambda: None)
    monkeypatch.setattr(launcher, "configure", lambda: [])
    monkeypatch.setitem(
        sys.modules,
        "solhunter_zero.macos_setup",
        types.SimpleNamespace(ensure_tools=lambda non_interactive=True: fake_log_startup("Ensuring tools")),
    )
    monkeypatch.setitem(
        sys.modules,
        "solhunter_zero.bootstrap_utils",
        types.SimpleNamespace(ensure_venv=lambda arg: fake_log_startup("Ensuring venv")),
    )
    monkeypatch.setitem(
        sys.modules,
        "solhunter_zero.env_config",
        types.SimpleNamespace(configure_startup_env=lambda root: None),
    )
    device_stub = types.SimpleNamespace(initialize_gpu=lambda: None)
    monkeypatch.setitem(sys.modules, "solhunter_zero.device", device_stub)
    monkeypatch.setattr(sys.modules["solhunter_zero"], "device", device_stub, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "solhunter_zero.system",
        types.SimpleNamespace(set_rayon_threads=lambda: None),
    )
    monkeypatch.setattr("solhunter_zero.logging_utils.setup_logging", lambda *a, **k: None)

    # Pretend previous runs already succeeded so we log skip messages
    tools_ok = tmp_path / "tools_ok"
    venv_ok = tmp_path / "venv_ok"
    tools_ok.write_text("ok")
    venv_ok.write_text("ok")
    import solhunter_zero.cache_paths as cp
    monkeypatch.setattr(cp, "TOOLS_OK_MARKER", tools_ok)
    monkeypatch.setattr(cp, "VENV_OK_MARKER", venv_ok)
    launcher.FAST_MODE = True

    # Replace execvp with a stub that emits final log messages
    def fake_execvp(cmd, args):
        fake_log_startup("startup launched")
        fake_log_startup("SolHunter Zero launch complete – system ready.")

    monkeypatch.setattr(launcher.os, "execvp", fake_execvp)

    launcher.main([])

    records = [r for r in caplog.records if r.name == "startup"]

    expected = [
        lambda m: m in ("Fast mode: skipping ensure_tools", "Ensuring tools"),
        lambda m: m in ("Fast mode: skipping ensure_venv", "Ensuring venv"),
        lambda m: m.startswith("Virtual environment:"),
        lambda m: m == "startup launched",
        lambda m: m == "SolHunter Zero launch complete – system ready.",
    ]

    idx = 0
    for check in expected:
        found = False
        while idx < len(records):
            if check(records[idx].message):
                found = True
                idx += 1
                break
            idx += 1
        assert found, "Startup log missing or out of order"
