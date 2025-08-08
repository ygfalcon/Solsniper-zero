import os

import solhunter_zero.env_config as env_config


def test_prompt_for_jito_auth(tmp_path, monkeypatch):
    """Supplying a token should keep MEV bundles enabled and persist auth."""

    monkeypatch.delenv("USE_MEV_BUNDLES", raising=False)
    monkeypatch.delenv("JITO_AUTH", raising=False)

    (tmp_path / "config.toml").write_text("use_mev_bundles = true\n", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")

    import sys
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt='': "TOKEN")

    env_config.configure_environment(tmp_path)

    assert os.environ.get("JITO_AUTH") == "TOKEN"
    assert "JITO_AUTH=TOKEN" in env_path.read_text(encoding="utf-8")
    assert os.environ.get("USE_MEV_BUNDLES") == "true"


def test_disable_mev_bundles_when_missing_jito_auth(tmp_path, monkeypatch):
    """MEV bundles should be disabled when JITO_AUTH is absent."""

    # Ensure a clean environment
    monkeypatch.delenv("USE_MEV_BUNDLES", raising=False)
    monkeypatch.delenv("JITO_AUTH", raising=False)

    # Create minimal config enabling MEV bundles
    (tmp_path / "config.toml").write_text("use_mev_bundles = true\n", encoding="utf-8")
    # Empty env file
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")

    env_config.configure_environment(tmp_path)

    assert os.environ.get("USE_MEV_BUNDLES") == "false"
    # The value should be persisted to the env file
    assert "USE_MEV_BUNDLES=false" in env_path.read_text(encoding="utf-8")
    assert os.environ.get("JITO_AUTH") in (None, "")

