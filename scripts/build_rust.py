"""Helpers for building Rust components used by SolHunter Zero."""

from __future__ import annotations

from pathlib import Path
import platform
import subprocess
import shutil

ROOT = Path(__file__).resolve().parent.parent


def build_rust_component(
    name: str, cargo_path: Path, output: Path, *, target: str | None = None
) -> None:
    """Build a Rust component and ensure its artifact exists.

    When ``target`` is provided the required Rust target is ensured and used for
    the build. The compiled binary or library is copied to ``output`` when
    necessary and automatically codesigned on macOS. A ``RuntimeError`` is
    raised when the expected artifact cannot be located after the build
    completes.
    """

    cmd = ["cargo", "build", "--manifest-path", str(cargo_path), "--release"]
    if target is not None:
        try:
            installed_targets = subprocess.check_output(
                ["rustup", "target", "list", "--installed"], text=True
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("failed to verify rust targets") from exc
        if target not in installed_targets:
            subprocess.check_call(["rustup", "target", "add", target])
        cmd.extend(["--target", target])
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            installed_targets = subprocess.check_output(
                ["rustup", "target", "list", "--installed"], text=True
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("failed to verify rust targets") from exc
        if "aarch64-apple-darwin" not in installed_targets:
            subprocess.check_call(["rustup", "target", "add", "aarch64-apple-darwin"])
        cmd.extend(["--target", "aarch64-apple-darwin"])

    subprocess.check_call(cmd)

    artifact = output.name
    target_dirs = [cargo_path.parent / "target", ROOT / "target"]
    candidates: list[Path] = []
    for base in target_dirs:
        candidates.append(base / "release" / artifact)
        if target is not None:
            candidates.append(base / target / "release" / artifact)
        elif platform.system() == "Darwin" and platform.machine() == "arm64":
            candidates.append(base / "aarch64-apple-darwin" / "release" / artifact)

    built = next((p for p in candidates if p.exists()), None)
    if built is None:
        paths = ", ".join(str(p) for p in candidates)
        raise RuntimeError(f"failed to build {name}: expected {artifact} in {paths}")

    if built.resolve() != output.resolve():
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built, output)

    if not output.exists():
        raise RuntimeError(
            f"{name} build succeeded but {output} is missing. Please build manually."
        )

    if platform.system() == "Darwin":
        subprocess.check_call(["codesign", "--force", "--sign", "-", str(output)])


def ensure_route_ffi() -> None:
    """Ensure the ``route_ffi`` Rust library is built and copied locally."""

    libname = (
        "libroute_ffi.dylib" if platform.system() == "Darwin" else "libroute_ffi.so"
    )
    libpath = ROOT / "solhunter_zero" / libname
    if libpath.exists():
        return

    build_rust_component(
        "route_ffi",
        ROOT / "route_ffi" / "Cargo.toml",
        libpath,
    )


def ensure_depth_service() -> None:
    """Build the ``depth_service`` binary if missing."""

    bin_path = ROOT / "target" / "release" / "depth_service"
    if bin_path.exists():
        return

    target = "aarch64-apple-darwin" if platform.system() == "Darwin" else None
    try:
        build_rust_component(
            "depth_service",
            ROOT / "depth_service" / "Cargo.toml",
            bin_path,
            target=target,
        )
    except Exception as exc:  # pragma: no cover - build errors are rare
        hint = ""
        if platform.system() == "Darwin":
            hint = " Hint: run 'scripts/mac_setup.py' to install macOS build tools."
        print(f"Failed to build depth_service: {exc}.{hint}")
        raise SystemExit(1)


def ensure_rust_components() -> None:
    """Ensure all Rust components required by the project are present."""

    ensure_route_ffi()
    ensure_depth_service()
