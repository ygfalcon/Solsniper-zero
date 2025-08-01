from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.build_ext import build_ext as _build_ext
import subprocess
import os




def build_route_ffi(root: Path, out_dir: Path):
    """Compile the Rust FFI library and copy it to *out_dir* when missing."""
    lib_dst = out_dir / "libroute_ffi.so"
    if lib_dst.exists():
        return
    cmd = [
        "cargo",
        "build",
        "--manifest-path",
        str(root / "route_ffi" / "Cargo.toml"),
        "--release",
    ]
    if os.getenv("ROUTE_FFI_PARALLEL") == "1":
        cmd.append("--features=parallel")
    subprocess.run(cmd, check=True)
    lib_src = root / "route_ffi" / "target" / "release" / "libroute_ffi.so"
    if lib_src.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        lib_dst.write_bytes(lib_src.read_bytes())


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

        # build the FFI library and copy it into the source tree
        build_route_ffi(root, out_py)

        super().run()

        # ensure the library is copied to the build directory
        lib_src = root / "route_ffi" / "target" / "release" / "libroute_ffi.so"
        if lib_src.exists():
            build_dst = Path(self.build_lib) / "solhunter_zero" / "libroute_ffi.so"
            build_dst.parent.mkdir(parents=True, exist_ok=True)
            build_dst.write_bytes(lib_src.read_bytes())


class build_ext(_build_ext):
    def run(self):
        root = Path(__file__).parent
        build_route_ffi(root, Path(self.build_lib) / "solhunter_zero")
        super().run()

setup(cmdclass={"build_py": build_py, "build_ext": build_ext})
