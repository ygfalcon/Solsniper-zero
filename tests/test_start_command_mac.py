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
    mac_setup = scripts_dir / 'mac_setup.sh'
    mac_setup.write_text("""#!/usr/bin/env bash
set -e
mkdir -p /opt/homebrew/bin
cat >/opt/homebrew/bin/brew <<'EOF'
#!/usr/bin/env bash
if [ \"$1\" = shellenv ]; then
  echo 'export PATH="/opt/homebrew/bin:$PATH"'
fi
EOF
chmod +x /opt/homebrew/bin/brew
cat >/opt/homebrew/bin/rustup <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x /opt/homebrew/bin/rustup
cat >/opt/homebrew/bin/python3.11 <<'EOF'
#!/usr/bin/env bash
if [ \"$1\" = -V ]; then echo 'Python 3.11.0'; else echo "$PATH"; fi
EOF
chmod +x /opt/homebrew/bin/python3.11
""")
    mac_setup.chmod(0o755)
    (scripts_dir / 'rotate_logs.sh').write_text('rotate_logs() { :; }\n')

    fakebin = tmp_path / 'fakebin'
    fakebin.mkdir()
    python3 = fakebin / 'python3'
    python3.write_text("""#!/usr/bin/env bash
if [ \"$1\" = -V ]; then
  echo 'Python 3.11.0'
else
  echo "$PATH"
fi
""")
    python3.chmod(0o755)
    os.symlink(python3, fakebin / 'python3.11')
    uname = fakebin / 'uname'
    uname.write_text("""#!/usr/bin/env bash
echo Darwin
""")
    uname.chmod(0o755)

    env = os.environ.copy()
    env['PATH'] = f"{fakebin}:{env['PATH']}"

    result = subprocess.run(['bash', str(start_cmd), '--skip-preflight'], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.stdout  # ensure script produced output
