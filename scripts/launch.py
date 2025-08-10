from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.progress import Progress

from solhunter_zero.bootstrap_utils import ensure_deps as _ensure_deps

__all__ = ["install_dependencies"]

console = Console()


def ensure_route_ffi() -> None:
    from solhunter_zero.build_utils import ensure_route_ffi as _ensure_route_ffi

    _ensure_route_ffi()


def ensure_depth_service() -> None:
    from solhunter_zero.build_utils import ensure_depth_service as _ensure_depth_service

    _ensure_depth_service()


def ensure_protos() -> None:
    from solhunter_zero.build_utils import ensure_protos as _ensure_protos

    _ensure_protos()


def install_dependencies(*, install_optional: bool = False, ensure_deps_func=_ensure_deps) -> None:
    """Install core dependencies and build required components."""
    with Progress(console=console, transient=True) as progress:
        with ThreadPoolExecutor() as executor:
            task_map = {
                executor.submit(
                    ensure_deps_func, install_optional=install_optional
                ): progress.add_task("Installing dependencies...", total=1),
                executor.submit(ensure_protos): progress.add_task(
                    "Generating protos...", total=1
                ),
                executor.submit(ensure_route_ffi): progress.add_task(
                    "Building route FFI...", total=1
                ),
                executor.submit(ensure_depth_service): progress.add_task(
                    "Building depth service...", total=1
                ),
            }
            for future in as_completed(task_map):
                progress.advance(task_map[future])
    console.print("[green]Dependencies installed[/]")
