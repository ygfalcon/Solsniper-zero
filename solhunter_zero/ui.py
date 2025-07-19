import threading
import time
import os
import json
from flask import Flask, jsonify, request

from . import wallet
from . import main as main_module
from .memory import Memory
from .portfolio import Portfolio

app = Flask(__name__)

trading_thread = None
stop_event = threading.Event()
loop_delay = 60

# currently active portfolio and keypair used by the trading loop
current_portfolio: Portfolio | None = None
current_keypair = None

def trading_loop() -> None:
    global current_portfolio, current_keypair
    memory = Memory("sqlite:///memory.db")
    portfolio = Portfolio()

    current_portfolio = portfolio
    keypair_path = os.getenv("KEYPAIR_PATH")
    keypair = wallet.load_keypair(keypair_path) if keypair_path else None
    current_keypair = keypair

    while not stop_event.is_set():
        keypair = wallet.load_selected_keypair()
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



@app.route("/balances")
def balances() -> dict:
    """Return the portfolio balances as JSON."""
    if current_portfolio is None:
        return jsonify({})
    data = {
        token: {"amount": pos.amount, "entry_price": pos.entry_price}
        for token, pos in current_portfolio.balances.items()
    }
    return jsonify(data)


HTML_PAGE = """
<!doctype html>
<html>
<head>
    <title>SolHunter UI</title>
</head>
<body>
    <button id='start'>Start</button>
    <button id='stop'>Stop</button>

    <table id='balances'>
        <thead><tr><th>Token</th><th>Amount</th></tr></thead>
        <tbody></tbody>
    </table>

    <script>
    document.getElementById('start').onclick = function() {
        fetch('/start', {method: 'POST'}).then(r => r.json()).then(console.log);
    };
    document.getElementById('stop').onclick = function() {
        fetch('/stop', {method: 'POST'}).then(r => r.json()).then(console.log);
    };


    function loadBalances() {
        fetch('/balances').then(r => r.json()).then(data => {
            const tbody = document.querySelector('#balances tbody');
            tbody.innerHTML = '';
            Object.entries(data).forEach(([token, info]) => {
                const row = document.createElement('tr');
                const t = document.createElement('td');
                t.textContent = token;
                const a = document.createElement('td');
                a.textContent = info.amount ?? info;
                row.appendChild(t);
                row.appendChild(a);
                tbody.appendChild(row);
            });
        });
    }
    loadBalances();
    setInterval(loadBalances, 5000);

    </script>
</body>
</html>
"""

@app.route("/")
def index() -> str:
    return HTML_PAGE

if __name__ == "__main__":
    app.run()
