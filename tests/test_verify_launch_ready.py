import platform
from pathlib import Path

def setup_env(root: Path):
    (root / 'config.toml').write_text('')
    kp = root / 'keypairs'
    kp.mkdir()
    (kp / 'active').write_text('default')
    (kp / 'default.json').write_text('{}')
    libname = 'libroute_ffi.dylib' if platform.system() == 'Darwin' else 'libroute_ffi.so'
    pkg_dir = root / 'solhunter_zero'
    pkg_dir.mkdir()
    (pkg_dir / libname).write_text('')
    depth_dir = root / 'target' / 'release'
    depth_dir.mkdir(parents=True)
    (depth_dir / 'depth_service').write_text('')


def test_verify_launch_ready_success(tmp_path, monkeypatch):
    setup_env(tmp_path)
    from solhunter_zero import bootstrap
    monkeypatch.setattr(bootstrap, 'ROOT', tmp_path)
    monkeypatch.setattr(bootstrap.shutil, 'which', lambda cmd: '/usr/bin/cargo')
    ok, msg = bootstrap.verify_launch_ready()
    assert ok, msg


def test_verify_launch_ready_missing_config(tmp_path, monkeypatch):
    setup_env(tmp_path)
    (tmp_path / 'config.toml').unlink()
    from solhunter_zero import bootstrap
    monkeypatch.setattr(bootstrap, 'ROOT', tmp_path)
    monkeypatch.setattr(bootstrap.shutil, 'which', lambda cmd: '/usr/bin/cargo')
    ok, msg = bootstrap.verify_launch_ready()
    assert not ok
    assert 'config.toml' in msg
