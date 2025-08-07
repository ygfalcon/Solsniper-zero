from solhunter_zero import bootstrap, bootstrap_checks


def _patch_bootstrap(monkeypatch):
    monkeypatch.setattr(bootstrap, "ensure_venv", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "ensure_deps", lambda install_optional=False: None)
    monkeypatch.setattr(bootstrap, "ensure_keypair", lambda: None)
    monkeypatch.setattr(bootstrap, "ensure_config", lambda: None)
    monkeypatch.setattr(bootstrap, "ensure_route_ffi", lambda: None)
    monkeypatch.setattr(bootstrap, "ensure_depth_service", lambda: None)
    monkeypatch.setattr(bootstrap.wallet, "ensure_default_keypair", lambda: None)
    monkeypatch.setattr(bootstrap.device, "ensure_gpu_env", lambda: {})
    monkeypatch.setattr(bootstrap, "load_config", lambda: {})
    monkeypatch.setattr(bootstrap, "validate_config", lambda cfg: cfg)


def test_bootstrap_runs_checks(monkeypatch):
    _patch_bootstrap(monkeypatch)

    called = {}
    monkeypatch.setattr(bootstrap, "check_internet", lambda: called.setdefault("internet", True))
    monkeypatch.setattr(bootstrap, "ensure_rpc", lambda warn_only=False: called.setdefault("rpc", True))
    monkeypatch.setattr(bootstrap, "ensure_endpoints", lambda cfg: called.setdefault("endpoints", cfg))
    monkeypatch.setattr(bootstrap, "check_disk_space", lambda mb: called.setdefault("disk", mb))

    bootstrap.bootstrap()

    assert called == {
        "internet": True,
        "rpc": True,
        "endpoints": {},
        "disk": 1 << 30,
    }


def test_bootstrap_respects_skip_env(monkeypatch):
    _patch_bootstrap(monkeypatch)

    monkeypatch.setenv("SOLHUNTER_SKIP_INTERNET", "1")
    monkeypatch.setenv("SOLHUNTER_SKIP_RPC", "1")
    monkeypatch.setenv("SOLHUNTER_SKIP_ENDPOINTS", "1")
    monkeypatch.setenv("SOLHUNTER_SKIP_DISK", "1")

    monkeypatch.setattr(bootstrap, "check_internet", lambda: (_ for _ in ()).throw(Exception("internet")))
    monkeypatch.setattr(bootstrap, "ensure_rpc", lambda warn_only=False: (_ for _ in ()).throw(Exception("rpc")))
    monkeypatch.setattr(bootstrap, "ensure_endpoints", lambda cfg: (_ for _ in ()).throw(Exception("endpoints")))
    monkeypatch.setattr(bootstrap, "check_disk_space", lambda mb: (_ for _ in ()).throw(Exception("disk")))

    bootstrap.bootstrap()

