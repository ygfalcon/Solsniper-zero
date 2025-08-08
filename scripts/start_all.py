import os
import sys
import signal
import threading
import asyncio

from solhunter_zero.paths import ROOT
from solhunter_zero.logging_utils import setup_logging, log_startup
from solhunter_zero import env
from solhunter_zero.http import close_session
from solhunter_zero.service_launcher import (
    start_background_services,
    stop_background_services,
)
import solhunter_zero.ui as ui


def main() -> None:
    setup_logging("startup")
    env.load_env_file(ROOT / ".env")
    os.chdir(ROOT)
    log_startup("start_all launched")

    if len(sys.argv) > 1 and sys.argv[1] == "autopilot":
        from solhunter_zero import autopilot

        autopilot.main()
        return

    services = start_background_services("config.toml")
    app = ui.create_app()

    def _shutdown(*_: object) -> None:
        stop_background_services(services)
        ui.stop_event.set()
        if ui.trading_thread:
            ui.trading_thread.join()
        asyncio.run(close_session())
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if ui.websockets is not None:
        def _start_rl_ws() -> None:
            ui.rl_ws_loop = asyncio.new_event_loop()
            ui.rl_ws_loop.run_until_complete(
                ui.websockets.serve(
                    ui._rl_ws_handler,
                    "localhost",
                    8767,
                    ping_interval=ui._WS_PING_INTERVAL,
                    ping_timeout=ui._WS_PING_TIMEOUT,
                )
            )
            ui.rl_ws_loop.run_forever()

        def _start_event_ws() -> None:
            ui.event_ws_loop = asyncio.new_event_loop()
            ui.event_ws_loop.run_until_complete(
                ui.websockets.serve(
                    ui._event_ws_handler,
                    "localhost",
                    8766,
                    path="/ws",
                    ping_interval=ui._WS_PING_INTERVAL,
                    ping_timeout=ui._WS_PING_TIMEOUT,
                )
            )
            ui.event_ws_loop.run_forever()

        def _start_log_ws() -> None:
            ui.log_ws_loop = asyncio.new_event_loop()
            ui.log_ws_loop.run_until_complete(
                ui.websockets.serve(
                    ui._log_ws_handler,
                    "localhost",
                    8768,
                    ping_interval=ui._WS_PING_INTERVAL,
                    ping_timeout=ui._WS_PING_TIMEOUT,
                )
            )
            ui.log_ws_loop.run_forever()

        threading.Thread(target=_start_rl_ws, daemon=True).start()
        threading.Thread(target=_start_event_ws, daemon=True).start()
        threading.Thread(target=_start_log_ws, daemon=True).start()

    ui.stop_event.clear()
    ui.trading_thread = threading.Thread(
        target=lambda: asyncio.run(ui.trading_loop()), daemon=True
    )
    ui.trading_thread.start()

    try:
        app.run()
    finally:
        _shutdown()


if __name__ == "__main__":
    main()

