import os
import subprocess
from pathlib import Path


def test_mac_setup_failure_propagates(tmp_path):
    repo_start = Path('start.command').read_text()
    start_cmd = tmp_path / 'start.command'
    start_cmd.write_text(repo_start)
    start_cmd.chmod(0o755)

    scripts_dir = tmp_path / 'scripts'
    scripts_dir.mkdir()
    mac_setup = scripts_dir / 'mac_setup.sh'
    mac_setup.write_text("""#!/usr/bin/env bash
exit 5
""")
    mac_setup.chmod(0o755)

    fakebin = tmp_path / 'fakebin'
    fakebin.mkdir()
    python3 = fakebin / 'python3'
    python3.write_text("""#!/usr/bin/env bash
if [ "$1" = -V ]; then
  echo 'Python 3.10.0'
fi
""")
    python3.chmod(0o755)

    env = os.environ.copy()
    env['PATH'] = f"{fakebin}:{env['PATH']}"

    result = subprocess.run(['bash', str(start_cmd)], cwd=tmp_path, env=env, capture_output=True, text=True)

    assert result.returncode == 5
    log_path = tmp_path / 'startup.log'
    assert log_path.exists()
    assert str(log_path) in result.stdout

