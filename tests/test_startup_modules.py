import subprocess
import urllib.request
import contextlib
import shutil
import sys
import types

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

import scripts.checks as checks
import scripts.wallet_setup as wallet_setup
import scripts.launch as launch


def test_disk_space_required_bytes_int():
    assert isinstance(checks.disk_space_required_bytes(), int)


def test_ensure_rpc_success(monkeypatch):
    class DummyResp:
        def read(self):
            return b"ok"

    def fake_urlopen(req, timeout=5):
        return contextlib.nullcontext(DummyResp())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    checks.ensure_rpc()


def test_ensure_wallet_cli_attempts_install(monkeypatch):
    calls = {}
    which_results = [None, "/usr/bin/solhunter-wallet"]

    monkeypatch.setattr(shutil, "which", lambda _: which_results.pop(0))

    def fake_run(cmd, text=True):
        calls["run"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    wallet_setup.ensure_wallet_cli()
    assert "run" in calls


def test_install_dependencies_invokes_steps(monkeypatch):
    called = {}

    def mark(name):
        called[name] = called.get(name, 0) + 1

    monkeypatch.setattr(launch, "ensure_route_ffi", lambda: mark("route"))
    monkeypatch.setattr(launch, "ensure_depth_service", lambda: mark("depth"))
    monkeypatch.setattr(launch, "ensure_protos", lambda: mark("protos"))

    def fake_ensure_deps(install_optional=False):
        mark("deps")

    launch.install_dependencies(ensure_deps_func=fake_ensure_deps)
    assert called == {"deps": 1, "protos": 1, "route": 1, "depth": 1}
