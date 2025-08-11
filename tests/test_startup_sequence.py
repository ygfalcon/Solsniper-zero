"""Verify launcher startup sequence with stubbed prerequisites."""

from __future__ import annotations

import os
import sys

import pytest


def test_startup_sequence(monkeypatch, capsys, stub_startup_prereqs, stub_run_first_trade):
    """Ensure launcher invokes expected startup steps in order."""

    # Avoid writing logs during tests
    monkeypatch.setenv("SOLHUNTER_TESTING", "1")
    import solhunter_zero.logging_utils as lu

    monkeypatch.setattr(lu, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(lu, "log_startup", lambda *a, **k: None)

    # Use current interpreter
    import solhunter_zero.python_env as pyenv

    monkeypatch.setattr(pyenv, "find_python", lambda repair=False: sys.executable)

    from scripts import startup as startup_script

    def fake_execvp(prog, argv):
        # argv: [python, '-m', 'scripts.startup', *args]
        code = startup_script._main_impl(argv[3:])
        raise SystemExit(code)

    monkeypatch.setattr(os, "execvp", fake_execvp)

    from solhunter_zero import launcher

    args = ["--skip-preflight", "--non-interactive", "demo"]
    with pytest.raises(SystemExit) as exc:
        launcher.main(args)
    assert exc.value.code == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert out == [
        "ensure_tools called",
        "ensure_venv called",
        "configure_startup_env called",
        "set_rayon_threads called",
        "initialize_gpu called",
        "startup_runner.launch_only ['demo']",
    ]

    # stub_run_first_trade records the command for further assertions if needed
    assert stub_run_first_trade["cmd"] == ["demo"]

