import threading
import time
import os
import json
import asyncio
from flask import Flask, jsonify, request

from .prices import fetch_token_prices

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
        asyncio.run(
            main_module._run_iteration(
                memory,
                portfolio,
                testnet=False,
                dry_run=False,
                offline=False,
                keypair=keypair,
            )
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
    """Return portfolio balances with USD values."""
    pf = Portfolio()
    tokens = list(pf.balances.keys())
    prices = fetch_token_prices(tokens)
    result = {}
    for token, pos in pf.balances.items():
        price = prices.get(token)
        usd = pos.amount * price if price is not None else None
        result[token] = {"amount": pos.amount, "price": price, "usd": usd}
    return jsonify(result)


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

        <thead><tr><th>Token</th><th>Amount</th><th>USD Value</th></tr></thead>

        <tbody></tbody>
    </table>

    <script>
    function refresh() {
        fetch('/balances').then(r => r.json()).then(data => {
            const body = document.querySelector('#balances tbody');
            body.innerHTML = '';
            Object.entries(data).forEach(([token, info]) => {
                const row = document.createElement('tr');
                row.innerHTML = `<td>${token}</td><td>${info.amount}</td><td>${info.usd}</td>`;
                body.appendChild(row);
            });
        });
    }
    document.getElementById('start').onclick = function() {
        fetch('/start', {method: 'POST'}).then(r => r.json()).then(console.log);
    };
    document.getElementById('stop').onclick = function() {
        fetch('/stop', {method: 'POST'}).then(r => r.json()).then(console.log);
    };

    refresh();

    </script>
</body>
</html>
"""

@app.route("/")
def index() -> str:
    return HTML_PAGE

if __name__ == "__main__":
    app.run()
