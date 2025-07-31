import json
import os
import ctypes

LIB = None

_libname = os.environ.get("ROUTE_FFI_LIB")
if _libname is None:
    _libname = os.path.join(os.path.dirname(__file__), "libroute_ffi.so")
if os.path.exists(_libname):
    try:
        LIB = ctypes.CDLL(_libname)
        LIB.best_route_json.argtypes = [
            ctypes.c_char_p,  # prices
            ctypes.c_char_p,  # fees
            ctypes.c_char_p,  # gas
            ctypes.c_char_p,  # latency
            ctypes.c_double,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_double),
        ]
        LIB.best_route_json.restype = ctypes.c_void_p
        LIB.free_cstring.argtypes = [ctypes.c_void_p]
    except OSError:
        LIB = None


def available() -> bool:
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
