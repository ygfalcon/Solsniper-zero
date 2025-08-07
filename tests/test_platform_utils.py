import pytest
from solhunter_zero import platform_utils


@pytest.mark.parametrize(
    "system,machine,is_arm64,needs_rosetta",
    [
        ("Darwin", "arm64", True, False),
        ("Darwin", "x86_64", False, True),
        ("Linux", "x86_64", False, False),
    ],
)
def test_platform_helpers(system, machine, is_arm64, needs_rosetta, monkeypatch):
    monkeypatch.setattr(platform_utils, "system", lambda: system)
    monkeypatch.setattr(platform_utils, "machine", lambda: machine)
    assert platform_utils.is_macos_arm64() is is_arm64
    assert platform_utils.requires_rosetta() is needs_rosetta
    assert platform_utils.system() == system
    assert platform_utils.machine() == machine
