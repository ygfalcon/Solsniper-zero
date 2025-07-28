from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
import subprocess

class build_py(_build_py):
    def run(self):
        proto_dir = Path(__file__).parent / "proto"
        out_py = Path(__file__).parent / "solhunter_zero"
        if proto_dir.exists():
            subprocess.run([
                "python",
                "-m",
                "grpc_tools.protoc",
                f"-I{proto_dir}",
                f"--python_out={out_py}",
                str(proto_dir / "event.proto"),
            ], check=True)
        super().run()

setup(cmdclass={"build_py": build_py})
