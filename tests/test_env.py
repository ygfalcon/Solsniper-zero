import os

from solhunter_zero import env


def test_load_env_file(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "ROOT", tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\nBAZ=qux\n# comment\n")

    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAZ", raising=False)

    env.load_env_file(env_file)
    assert os.environ["FOO"] == "bar"
    assert os.environ["BAZ"] == "qux"

    monkeypatch.setenv("FOO", "orig")
    monkeypatch.setenv("BAZ", "orig2")
    env.load_env_file(env_file)
    assert os.environ["FOO"] == "orig"
    assert os.environ["BAZ"] == "orig2"


def test_creates_env_from_template(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "ROOT", tmp_path)
    target = tmp_path / ".env"
    template = tmp_path / ".env.example"
    template.write_text("NEW=val\n")
    monkeypatch.delenv("NEW", raising=False)
    env.load_env_file(target)
    assert target.exists()
    assert os.environ["NEW"] == "val"
    assert target.read_text() == "NEW=val\n"
    assert (target.stat().st_mode & 0o777) == 0o600


def test_creates_empty_env_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "ROOT", tmp_path)
    target = tmp_path / ".env"
    env.load_env_file(target)
    assert target.exists()
    assert target.read_text() == ""
    assert (target.stat().st_mode & 0o777) == 0o600
