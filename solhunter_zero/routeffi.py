import json
import os
import ctypes
import subprocess
from pathlib import Path

LIB = None

_libname = os.environ.get("ROUTE_FFI_LIB")
if _libname is None:
    _libname = os.path.join(os.path.dirname(__file__), "libroute_ffi.so")


def _load_lib() -> bool:
    """Attempt to load the FFI library and build it when missing."""
    global LIB

    if os.path.exists(_libname):
        try:
            LIB = ctypes.CDLL(_libname)
        except OSError:
            LIB = None
    else:
        manifest = Path(__file__).resolve().parents[1] / "route_ffi/Cargo.toml"
        if manifest.exists():
            try:
                subprocess.run([
                    "cargo",
                    "build",
                    "--manifest-path",
                    str(manifest),
                    "--release",
                ], check=True)
            except Exception:
                return False
        if os.path.exists(_libname):
            try:
                LIB = ctypes.CDLL(_libname)
            except OSError:
                LIB = None
    if LIB is not None:
        LIB.best_route_json.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_double,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_double),
        ]
        LIB.best_route_json.restype = ctypes.c_void_p
        LIB.free_cstring.argtypes = [ctypes.c_void_p]
        os.environ.setdefault("USE_FFI_ROUTE", "1")
    return LIB is not None


_load_lib()


def available() -> bool:
    if LIB is None:
        _load_lib()
    return LIB is not None


def best_route(prices: dict[str, float], amount: float, *, fees=None, gas=None, latency=None, max_hops=4):
    if LIB is None:
        return None
    prof = ctypes.c_double()
    prices_json = json.dumps(prices).encode()
    fees_json = json.dumps(fees or {}).encode()
    gas_json = json.dumps(gas or {}).encode()
    lat_json = json.dumps(latency or {}).encode()
    ptr = LIB.best_route_json(prices_json, fees_json, gas_json, lat_json, amount, max_hops, ctypes.byref(prof))
    if not ptr:
        return None
    path_json = ctypes.string_at(ptr).decode()
    LIB.free_cstring(ptr)
    path = json.loads(path_json)
    return path, prof.value
