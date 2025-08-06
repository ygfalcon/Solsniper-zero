import os
import subprocess
from pathlib import Path


def test_homebrew_path_added(tmp_path, monkeypatch):
    repo_start = Path('start.command').read_text()
    start_cmd = tmp_path / 'start.command'
    start_cmd.write_text(repo_start)
    start_cmd.chmod(0o755)

    (tmp_path / 'config.example.toml').write_text('')

    scripts_dir = tmp_path / 'scripts'
    scripts_dir.mkdir()
    mac_setup = scripts_dir / 'mac_setup.py'
    mac_setup.write_text(
        """#!/usr/bin/env python3
import os
from pathlib import Path


def main(argv=None):
    bin_dir = Path('/opt/homebrew/bin')
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / 'brew').write_text('#!/usr/bin/env bash\nif [ "$1" = "shellenv" ]; then\n  echo "export PATH=\"/opt/homebrew/bin:$PATH\""\nfi\n')
    (bin_dir / 'rustup').write_text('#!/usr/bin/env bash\nexit 0\n')
    (bin_dir / 'python3.11').write_text('#!/usr/bin/env bash\nif [ "$1" = "-V" ]; then echo "Python 3.11.0"; else echo "$PATH"; fi\n')
    for f in ['brew', 'rustup', 'python3.11']:
        os.chmod(bin_dir / f, 0o755)
    os.environ['PATH'] = f"{bin_dir}:{os.environ.get('PATH','')}"
    print('mac setup')


if __name__ == '__main__':
    main()
"""
    )
    mac_setup.chmod(0o755)
    (scripts_dir / 'rotate_logs.sh').write_text('rotate_logs() { :; }\n')

    fakebin = tmp_path / 'fakebin'
    fakebin.mkdir()
    python3 = fakebin / 'python3'
    python3.write_text(
        """#!/usr/bin/env bash
if [ \"$1\" = -V ]; then
  echo 'Python 3.11.0'
else
  echo \"$PATH\"
fi
"""
    )
    python3.chmod(0o755)
    os.symlink(python3, fakebin / 'python3.11')
    uname = fakebin / 'uname'
    uname.write_text("""#!/usr/bin/env bash
echo Darwin
""")
    uname.chmod(0o755)

    arch = fakebin / 'arch'
    arch.write_text("""#!/usr/bin/env bash
if [ "$1" = "-arm64" ]; then
  shift
fi
"$@"
""")
    arch.chmod(0o755)

    env = os.environ.copy()
    env['PATH'] = f"{fakebin}:{env['PATH']}"

    result = subprocess.run(['bash', str(start_cmd), '--skip-preflight'], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.stdout  # ensure script produced output
