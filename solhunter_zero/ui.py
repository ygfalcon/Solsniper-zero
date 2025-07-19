import threading
import time
import os
from flask import Flask, jsonify

from . import wallet
from . import main as main_module
from .memory import Memory
from .portfolio import Portfolio

app = Flask(__name__)

trading_thread = None
stop_event = threading.Event()
loop_delay = 60

def trading_loop() -> None:
    memory = Memory("sqlite:///memory.db")
    portfolio = Portfolio()
    keypair_path = os.getenv("KEYPAIR_PATH")
    keypair = wallet.load_keypair(keypair_path) if keypair_path else None
    while not stop_event.is_set():
        main_module._run_iteration(
            memory,
            portfolio,
            testnet=False,
            dry_run=False,
            offline=False,
            keypair=keypair,
        )
        for _ in range(loop_delay):
            if stop_event.is_set():
                break
            time.sleep(1)

@app.route("/start", methods=["POST"])
def start() -> dict:
    global trading_thread
    if trading_thread and trading_thread.is_alive():
        return jsonify({"status": "already running"})
    stop_event.clear()
    trading_thread = threading.Thread(target=trading_loop, daemon=True)
    trading_thread.start()
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop() -> dict:
    stop_event.set()
    if trading_thread:
        trading_thread.join()
    return jsonify({"status": "stopped"})

HTML_PAGE = """
<!doctype html>
<html>
<head>
    <title>SolHunter UI</title>
</head>
<body>
    <button id='start'>Start</button>
    <button id='stop'>Stop</button>
    <script>
    document.getElementById('start').onclick = function() {
        fetch('/start', {method: 'POST'}).then(r => r.json()).then(console.log);
    };
    document.getElementById('stop').onclick = function() {
        fetch('/stop', {method: 'POST'}).then(r => r.json()).then(console.log);
    };
    </script>
</body>
</html>
"""

@app.route("/")
def index() -> str:
    return HTML_PAGE

if __name__ == "__main__":
    app.run()
