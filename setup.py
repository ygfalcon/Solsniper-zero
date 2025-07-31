from pathlib import Path
import shutil
import subprocess
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

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
        ffi_dir = Path(__file__).parent / "route_ffi"
        if ffi_dir.exists():
            subprocess.run([
                "cargo",
                "build",
                "--manifest-path",
                str(ffi_dir / "Cargo.toml"),
                "--release",
            ], check=True)
            built = ffi_dir / "target/release/libroute_ffi.so"
            if built.exists():
                shutil.copy2(built, out_py / "libroute_ffi.so")
        super().run()

setup(cmdclass={"build_py": build_py})
