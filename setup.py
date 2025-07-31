from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
import subprocess

class build_py(_build_py):
    def run(self):
        root = Path(__file__).parent
        proto_dir = root / "proto"
        out_py = root / "solhunter_zero"
        if proto_dir.exists():
            subprocess.run([
                "python",
                "-m",
                "grpc_tools.protoc",
                f"-I{proto_dir}",
                f"--python_out={out_py}",
                str(proto_dir / "event.proto"),
            ], check=True)

        # build the FFI library
        subprocess.run([
            "cargo",
            "build",
            "--manifest-path",
            str(root / "route_ffi" / "Cargo.toml"),
            "--release",
        ], check=True)
        lib_src = root / "route_ffi" / "target" / "release" / "libroute_ffi.so"
        if lib_src.exists():
            lib_dst = out_py / "libroute_ffi.so"
            lib_dst.write_bytes(lib_src.read_bytes())

        super().run()

        # ensure the library is copied to the build directory
        if lib_src.exists():
            build_dst = Path(self.build_lib) / "solhunter_zero" / "libroute_ffi.so"
            build_dst.parent.mkdir(parents=True, exist_ok=True)
            build_dst.write_bytes(lib_src.read_bytes())

setup(cmdclass={"build_py": build_py})
