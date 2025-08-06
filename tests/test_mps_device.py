import os
import subprocess
from pathlib import Path

import torch

from solhunter_zero import device as device_module


def test_get_default_device_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert getattr(device_module.get_default_device(), "type", None) == "mps"


def _extract_has_gpu() -> str:
    run_sh = Path(__file__).resolve().parent.parent / "run.sh"
    lines = run_sh.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("has_gpu()"))
    func_lines = [lines[start]]
    depth = lines[start].count("{") - lines[start].count("}")
    for line in lines[start + 1:]:
        func_lines.append(line)
        depth += line.count("{") - line.count("}")
        if depth == 0:
            break
    return "\n".join(func_lines)


def test_run_sh_detects_mps(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uname = bin_dir / "uname"
    uname.write_text("#!/bin/sh\necho Darwin\n")
    uname.chmod(0o755)
    system_profiler = bin_dir / "system_profiler"
    system_profiler.write_text("#!/bin/sh\necho 'Vendor: Apple'\necho 'Metal: Supported'\n")
    system_profiler.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["GPU_VENDOR"] = "apple"

    script = _extract_has_gpu() + "\nif has_gpu; then echo yes; else echo no; fi"
    result = subprocess.run([
        "bash",
        "-c",
        script,
    ], capture_output=True, text=True, env=env, check=True)
    assert result.stdout.strip() == "yes"
