import os

from solhunter_zero.env_flags import env_flags


def test_env_flags_sets_and_restores(monkeypatch):
    monkeypatch.setenv("FOO", "bar")
    with env_flags(FOO="baz", BAZ="qux"):
        assert os.environ["FOO"] == "baz"
        assert os.environ["BAZ"] == "qux"
    assert os.environ["FOO"] == "bar"
    assert "BAZ" not in os.environ
