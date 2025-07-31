from .jsonutil import dumps, loads
import os
import ctypes
import logging

logger = logging.getLogger(__name__)

LIB = None

_libname = os.environ.get("ROUTE_FFI_LIB")
if _libname is None:
    _libname = os.path.join(os.path.dirname(__file__), "libroute_ffi.so")
if os.path.exists(_libname):
    try:
        LIB = ctypes.CDLL(_libname)
        _search = getattr(LIB, "search_route_json", None)
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
        if _search is not None:
            _search.argtypes = list(LIB.best_route_json.argtypes)
            _search.restype = ctypes.c_void_p
            LIB.search_route_json = _search
        _parallel = getattr(LIB, "best_route_parallel_json", None)
        if _parallel is not None:
            _parallel.argtypes = list(LIB.best_route_json.argtypes)
            _parallel.restype = ctypes.c_void_p
            LIB.best_route_parallel_json = _parallel
        _parallel_flag = getattr(LIB, "route_parallel_enabled", None)
        if _parallel_flag is not None:
            _parallel_flag.restype = ctypes.c_bool
            LIB.route_parallel_enabled = _parallel_flag
        _decode = getattr(LIB, "decode_token_agg_json", None)
        if _decode is not None:
            _decode.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            _decode.restype = ctypes.c_void_p
            LIB.decode_token_agg_json = _decode
        LIB.free_cstring.argtypes = [ctypes.c_void_p]
    except OSError:
        LIB = None


def available() -> bool:
    return LIB is not None


def parallel_enabled() -> bool:
    if LIB is None:
        return False
    flag = getattr(LIB, "route_parallel_enabled", None)
    if flag is None:
        return False
    return bool(flag())


def best_route(
    prices: dict[str, float],
    amount: float,
    *,
    fees=None,
    gas=None,
    latency=None,
    max_hops=4,
):
    if LIB is None:
        return None
    prof = ctypes.c_double()
    prices_json = dumps(prices).encode()
    fees_json = dumps(fees or {}).encode()
    gas_json = dumps(gas or {}).encode()
    lat_json = dumps(latency or {}).encode()
    func = getattr(LIB, "search_route_json", None)
    if func is None:
        func = LIB.best_route_json
    ptr = func(
        prices_json, fees_json, gas_json, lat_json, amount, max_hops, ctypes.byref(prof)
    )
    if not ptr:
        return None
    path_json = ctypes.string_at(ptr).decode()
    LIB.free_cstring(ptr)
    path = loads(path_json)
    return path, prof.value


def best_route_parallel(
    prices: dict[str, float],
    amount: float,
    *,
    fees=None,
    gas=None,
    latency=None,
    max_hops=4,
):
    if LIB is None:
        return None
    func = getattr(LIB, "best_route_parallel_json", None)
    if func is None:
        return best_route(
            prices,
            amount,
            fees=fees,
            gas=gas,
            latency=latency,
            max_hops=max_hops,
        )
    prof = ctypes.c_double()
    prices_json = dumps(prices).encode()
    fees_json = dumps(fees or {}).encode()
    gas_json = dumps(gas or {}).encode()
    lat_json = dumps(latency or {}).encode()
    ptr = func(
        prices_json,
        fees_json,
        gas_json,
        lat_json,
        amount,
        max_hops,
        ctypes.byref(prof),
    )
    if not ptr:
        return None
    path_json = ctypes.string_at(ptr).decode()
    LIB.free_cstring(ptr)
    path = loads(path_json)
    return path, prof.value


def decode_token_agg(buf: bytes) -> dict | None:
    if LIB is None:
        return None
    func = getattr(LIB, "decode_token_agg_json", None)
    if func is None:
        return None
    ptr = func(ctypes.c_char_p(buf), len(buf))
    if not ptr:
        return None
    s = ctypes.string_at(ptr).decode()
    LIB.free_cstring(ptr)
    return loads(s)


if LIB is not None and not parallel_enabled():
    logger.error(
        "Parallel route search unavailable. "
        "Rebuild route_ffi with 'cargo build --release --features=parallel'"
    )
