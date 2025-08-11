from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple, List

from rich.panel import Panel
from rich.console import Console

console = Console()


def render_banner() -> None:
    """Render the startup banner when running in a TTY."""
    if sys.stdout.isatty():
        console.print(Panel.fit("[bold cyan]SolHunter Zero Startup[/]"), justify="center")


def parse_args(argv: List[str] | None = None) -> Tuple[argparse.Namespace, List[str]]:
    """Parse command line arguments for startup."""
    parser = argparse.ArgumentParser(description="Guided setup and launch")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency check")
    parser.add_argument("--full-deps", action="store_true", help="Install optional dependencies")
    parser.add_argument("--skip-setup", action="store_true", help="Skip config and wallet prompts")
    parser.add_argument(
        "--skip-rpc-check",
        action="store_true",
        help="Skip Solana RPC availability check",
    )
    parser.add_argument(
        "--skip-endpoint-check",
        action="store_true",
        help="Skip HTTP endpoint availability check",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip internet and RPC connectivity checks",
    )
    parser.add_argument("--skip-preflight", action="store_true", help="Skip environment preflight checks")
    parser.add_argument("--self-test", action="store_true", help="Run bootstrap and preflight checks then exit")
    parser.add_argument("--one-click", action="store_true", help="Enable fully automated non-interactive startup")
    parser.set_defaults(one_click=True)
    parser.add_argument(
        "--allow-rosetta",
        action="store_true",
        help="Allow running under Rosetta (no Metal acceleration)",
    )
    parser.add_argument("--diagnostics", action="store_true", help="Print system diagnostics and exit")
    parser.add_argument("--no-diagnostics", action="store_true", help="Suppress post-run diagnostics collection")
    parser.add_argument("--repair", action="store_true", help="Force macOS setup and clear dependency caches")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip prompts and launch start_all.py directly",
    )
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--keypair", help="Path to Solana keypair file")
    args, rest = parser.parse_known_args(argv)

    if args.config:
        os.environ["SOLHUNTER_CONFIG"] = str(Path(args.config))
    if args.keypair:
        os.environ["KEYPAIR_PATH"] = str(Path(args.keypair))
    return args, rest


def launch_non_interactive(rest: List[str]) -> int:
    """Launch start_all.py directly for non-interactive runs."""
    proc = subprocess.run([sys.executable, "scripts/start_all.py", *rest])
    return proc.returncode
