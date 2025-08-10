import os
import subprocess
import sys
import types

import pytest


def get_startup(monkeypatch):
    crypto = types.ModuleType("cryptography")
    fernet = types.ModuleType("cryptography.fernet")
    fernet.Fernet = object
    fernet.InvalidToken = Exception
    crypto.fernet = fernet
    sys.modules.setdefault("cryptography", crypto)
    sys.modules.setdefault("cryptography.fernet", fernet)
    dummy_pydantic = types.SimpleNamespace(
        BaseModel=object,
        AnyUrl=str,
        ValidationError=Exception,
        root_validator=lambda *a, **k: (lambda f: f),
        validator=lambda *a, **k: (lambda f: f),
        field_validator=lambda *a, **k: (lambda f: f),
        model_validator=lambda *a, **k: (lambda f: f),
    )
    sys.modules.setdefault("pydantic", dummy_pydantic)
    class Console:
        def print(self, *a, **k):
            pass

    class Progress:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def add_task(self, *a, **k):
            return 1

        def advance(self, *a, **k):
            pass

    class Panel:
        @staticmethod
        def fit(*a, **k):
            return ""

    class Table:
        def __init__(self, *a, **k):
            pass

        def add_column(self, *a, **k):
            pass

        def add_row(self, *a, **k):
            pass

    rich = types.ModuleType("rich")
    rich_console = types.ModuleType("rich.console")
    rich_console.Console = Console
    rich_progress = types.ModuleType("rich.progress")
    rich_progress.Progress = Progress
    rich_panel = types.ModuleType("rich.panel")
    rich_panel.Panel = Panel
    rich_table = types.ModuleType("rich.table")
    rich_table.Table = Table
    rich.console = rich_console
    rich.progress = rich_progress
    rich.panel = rich_panel
    rich.table = rich_table
    sys.modules.setdefault("rich", rich)
    sys.modules.setdefault("rich.console", rich_console)
    sys.modules.setdefault("rich.progress", rich_progress)
    sys.modules.setdefault("rich.panel", rich_panel)
    sys.modules.setdefault("rich.table", rich_table)
    from scripts import startup

    return startup


def patch_minimal(monkeypatch, startup):
    monkeypatch.setattr(startup.preflight_utils, "check_disk_space", lambda r: (True, "ok"))
    monkeypatch.setattr(startup, "ensure_cargo", lambda: None)
    monkeypatch.setattr("solhunter_zero.bootstrap.bootstrap", lambda one_click=False: None)
    monkeypatch.setattr(startup.device, "initialize_gpu", lambda: {})
    monkeypatch.setattr(startup, "load_config", lambda *a, **k: {})
    monkeypatch.setattr("solhunter_zero.agent_manager.AgentManager.from_config", lambda *a, **k: object())
    monkeypatch.setattr(startup.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0))
    monkeypatch.setattr("scripts.healthcheck.main", lambda *a, **k: 0)
    monkeypatch.setattr(startup.preflight, "CHECKS", [])


def test_skip_venv_removed_after_success(monkeypatch):
    startup = get_startup(monkeypatch)
    patch_minimal(monkeypatch, startup)
    monkeypatch.delenv("SOLHUNTER_SKIP_VENV", raising=False)
    code = startup.main([
        "--skip-preflight",
        "--skip-deps",
        "--skip-setup",
        "--offline",
        "--skip-rpc-check",
        "--skip-endpoint-check",
        "--no-diagnostics",
    ])
    assert code == 0
    assert "SOLHUNTER_SKIP_VENV" not in os.environ


def test_skip_venv_restored_on_error(monkeypatch):
    startup = get_startup(monkeypatch)
    monkeypatch.setenv("SOLHUNTER_SKIP_VENV", "orig")

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(startup.preflight_utils, "check_disk_space", boom)
    with pytest.raises(RuntimeError):
        startup.main([
            "--skip-preflight",
            "--skip-deps",
            "--skip-setup",
            "--offline",
            "--skip-rpc-check",
            "--skip-endpoint-check",
            "--no-diagnostics",
        ])
    assert os.environ.get("SOLHUNTER_SKIP_VENV") == "orig"
