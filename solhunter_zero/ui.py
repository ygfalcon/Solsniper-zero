import threading
import time
import os
import asyncio
from flask import Flask, jsonify, request

from .config import load_config, apply_env_overrides, set_env_from_config

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

REQUIRED_ENV_VARS = ("BIRDEYE_API_KEY", "DEX_BASE_URL")


def _missing_required() -> list[str]:
    """Return names of required variables that are unset."""
    return [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]

def trading_loop() -> None:
    global current_portfolio, current_keypair

    cfg = apply_env_overrides(load_config("config.toml"))
    set_env_from_config(cfg)

    memory = Memory("sqlite:///memory.db")
    portfolio = Portfolio()

    current_portfolio = portfolio
    keypair_path = os.getenv("KEYPAIR_PATH")
    env_keypair = wallet.load_keypair(keypair_path) if keypair_path else None
    current_keypair = env_keypair

    while not stop_event.is_set():
        selected_keypair = wallet.load_selected_keypair()
        keypair = selected_keypair if selected_keypair is not None else env_keypair
        current_keypair = keypair
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

    cfg = apply_env_overrides(load_config("config.toml"))
    set_env_from_config(cfg)

    missing = _missing_required()
    if missing:
        msg = "Missing required configuration: " + ", ".join(missing)
        return jsonify({"status": "error", "message": msg}), 400

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


@app.route("/risk", methods=["GET", "POST"])
def risk_params() -> dict:
    if request.method == "POST":
        data = request.get_json() or {}
        rt = data.get("risk_tolerance")
        ma = data.get("max_allocation")
        rm = data.get("risk_multiplier")
        if rt is not None:
            os.environ["RISK_TOLERANCE"] = str(rt)
        if ma is not None:
            os.environ["MAX_ALLOCATION"] = str(ma)
        if rm is not None:
            os.environ["RISK_MULTIPLIER"] = str(rm)
        return jsonify({"status": "ok"})
    return jsonify(
        {
            "risk_tolerance": float(os.getenv("RISK_TOLERANCE", "0.1")),
            "max_allocation": float(os.getenv("MAX_ALLOCATION", "0.2")),
            "risk_multiplier": float(os.getenv("RISK_MULTIPLIER", "1.0")),
        }
    )



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

    <div id='risk'>
        <label>Risk tolerance <input id='risk_tolerance' type='number' step='0.01'></label>
        <label>Max allocation <input id='max_allocation' type='number' step='0.01'></label>
        <label>Risk multiplier <input id='risk_multiplier' type='number' step='0.01'></label>
        <button id='save_risk'>Save</button>
    </div>

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

    function loadRisk() {
        fetch('/risk').then(r => r.json()).then(data => {
            document.getElementById('risk_tolerance').value = data.risk_tolerance;
            document.getElementById('max_allocation').value = data.max_allocation;
            document.getElementById('risk_multiplier').value = data.risk_multiplier;
        });
    }
    document.getElementById('save_risk').onclick = function() {
        const data = {
            risk_tolerance: parseFloat(document.getElementById('risk_tolerance').value),
            max_allocation: parseFloat(document.getElementById('max_allocation').value),
            risk_multiplier: parseFloat(document.getElementById('risk_multiplier').value)
        };
        fetch('/risk', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)}).then(r => r.json()).then(console.log);
    };

    refresh();
    loadRisk();

    </script>
</body>
</html>
"""

@app.route("/")
def index() -> str:
    return HTML_PAGE

if __name__ == "__main__":
    app.run()
