#!/usr/bin/env python3
"""Interactive startup script for SolHunter Zero."""
from __future__ import annotations

import sys

from solhunter_zero import startup_cli, startup_checks, startup_runner
from solhunter_zero.logging_utils import log_startup, rotate_preflight_log
from solhunter_zero.config import apply_env_overrides, load_config
from solhunter_zero.bootstrap_utils import ensure_deps, ensure_endpoints
from solhunter_zero.rpc_utils import ensure_rpc
from solhunter_zero import device  # noqa: F401
from scripts import preflight  # noqa: F401
from scripts import deps  # noqa: F401
import solhunter_zero.bootstrap_utils as bootstrap_utils  # noqa: F401

# Re-export helpers for tests and external users
ensure_target = startup_checks.ensure_target
ensure_wallet_cli = startup_checks.ensure_wallet_cli
run_quick_setup = startup_checks.run_quick_setup
ensure_cargo = startup_checks.ensure_cargo

log_startup("startup launched")
rotate_preflight_log()


def main(argv: list[str] | None = None) -> int:
    args, rest = startup_cli.parse_args(argv)
    if args.non_interactive:
        return startup_runner.launch_only(rest)
    ctx = startup_checks.perform_checks(
        args,
        rest,
        ensure_deps=ensure_deps,
        ensure_target=ensure_target,
        ensure_wallet_cli=ensure_wallet_cli,
        ensure_rpc=ensure_rpc,
        ensure_endpoints=ensure_endpoints,
        ensure_cargo=ensure_cargo,
        run_quick_setup=run_quick_setup,
        log_startup=log_startup,
        apply_env_overrides=apply_env_overrides,
        load_config=load_config,
    )
    code = ctx.pop("code", 0)
    if code:
        return code
    return startup_runner.run(args, ctx, log_startup=log_startup)


def run(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    try:
        exit_code = main(args_list)
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1
    except Exception:
        if "--no-diagnostics" not in args_list:
            from scripts import diagnostics

            diagnostics.main()
        raise
    if exit_code and "--no-diagnostics" not in args_list:
        from scripts import diagnostics

        diagnostics.main()
    return exit_code or 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
