import threading
import time
import os
import asyncio
import json
from flask import Flask, jsonify, request
from pathlib import Path

from .config import load_config, apply_env_overrides, set_env_from_config

from .config import load_config, apply_env_overrides, set_env_from_config

from .prices import fetch_token_prices

from . import wallet
from . import main as main_module
from .memory import Memory
from .portfolio import Portfolio
from .config import (
    list_configs,
    save_config,
    select_config,
    get_active_config_name,
    set_env_from_config,
    load_selected_config,
)

_DEFAULT_PRESET = Path(__file__).resolve().parent.parent / "config.highrisk.toml"
cfg = load_config()
if not cfg and _DEFAULT_PRESET.is_file():
    cfg = load_config(_DEFAULT_PRESET)
cfg = apply_env_overrides(cfg)
set_env_from_config(cfg)

app = Flask(__name__)

trading_thread = None
stop_event = threading.Event()
loop_delay = 60

# currently active portfolio and keypair used by the trading loop
current_portfolio: Portfolio | None = None
current_keypair = None

# ``BIRDEYE_API_KEY`` is optional when ``SOLANA_RPC_URL`` is provided for
# on-chain scanning.
REQUIRED_ENV_VARS = ("DEX_BASE_URL",)


def _missing_required() -> list[str]:
    """Return names of required variables that are unset."""
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if not (os.getenv("BIRDEYE_API_KEY") or os.getenv("SOLANA_RPC_URL")):
        missing.append("BIRDEYE_API_KEY or SOLANA_RPC_URL")
    return missing

def trading_loop() -> None:
    global current_portfolio, current_keypair

    cfg = apply_env_overrides(load_config("config.toml"))
    set_env_from_config(cfg)

    memory = Memory("sqlite:///memory.db")
    portfolio = Portfolio()

    current_portfolio = portfolio
    set_env_from_config(load_selected_config())
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


@app.route("/discovery", methods=["GET", "POST"])
def discovery_method() -> dict:
    if request.method == "POST":
        method = (request.get_json() or {}).get("method")
        if method:
            os.environ["DISCOVERY_METHOD"] = method
        return jsonify({"status": "ok"})
    return jsonify({"method": os.getenv("DISCOVERY_METHOD", "websocket")})


@app.route("/keypairs", methods=["GET"])
def keypairs() -> dict:
    return jsonify(
        {
            "keypairs": wallet.list_keypairs(),
            "active": wallet.get_active_keypair_name(),
        }
    )


@app.route("/keypairs/upload", methods=["POST"])
def upload_keypair() -> dict:
    file = request.files.get("file")
    name = request.form.get("name") or (file.filename if file else None)
    if not file or not name:
        return jsonify({"error": "missing file or name"}), 400
    data = file.read()
    try:
        wallet.save_keypair(name, list(json.loads(data)))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "ok"})


@app.route("/keypairs/select", methods=["POST"])
def select_keypair_route() -> dict:
    name = request.get_json().get("name")
    wallet.select_keypair(name)
    return jsonify({"status": "ok"})


@app.route("/configs", methods=["GET"])
def configs() -> dict:
    return jsonify({"configs": list_configs(), "active": get_active_config_name()})


@app.route("/configs/upload", methods=["POST"])
def upload_config() -> dict:
    file = request.files.get("file")
    name = request.form.get("name") or (file.filename if file else None)
    if not file or not name:
        return jsonify({"error": "missing file or name"}), 400
    try:
        save_config(name, file.read())
    except Exception as exc:  # mirror keypair upload behavior
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "ok"})


@app.route("/configs/select", methods=["POST"])
def select_config_route() -> dict:
    name = request.get_json().get("name")
    select_config(name)
    return jsonify({"status": "ok"})


@app.route("/positions")
def positions() -> dict:
    pf = current_portfolio or Portfolio()
    tokens = list(pf.balances.keys())
    prices = fetch_token_prices(tokens)
    result = {}
    for token, pos in pf.balances.items():
        price = prices.get(token, pos.entry_price)
        roi = pf.position_roi(token, price)
        result[token] = {
            "amount": pos.amount,
            "entry_price": pos.entry_price,
            "current_price": price,
            "roi": roi,
        }
    return jsonify(result)


@app.route("/trades")
def trades() -> dict:
    mem = Memory("sqlite:///memory.db")
    recents = [
        {
            "token": t.token,
            "direction": t.direction,
            "amount": t.amount,
            "price": t.price,
            "timestamp": t.timestamp.isoformat(),
        }
        for t in mem.list_trades()[-50:]
    ]
    return jsonify(recents)


@app.route("/roi")
def roi() -> dict:
    pf = current_portfolio or Portfolio()
    tokens = list(pf.balances.keys())
    prices = fetch_token_prices(tokens)
    entry = sum(p.amount * p.entry_price for p in pf.balances.values())
    value = sum(p.amount * prices.get(tok, p.entry_price) for tok, p in pf.balances.items())
    roi = (value - entry) / entry if entry else 0.0
    return jsonify({"roi": roi})



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

    <div id='keys'>
        <h3>Keypair</h3>
        <select id='keypair_select'></select>
        <input type='file' id='keypair_file'>
        <input type='text' id='keypair_name' placeholder='Name'>
        <button id='upload_keypair'>Upload</button>
    </div>

    <div id='configs'>
        <h3>Config</h3>
        <select id='config_select'></select>
        <input type='file' id='config_file'>
        <input type='text' id='config_name' placeholder='Name'>
        <button id='upload_config'>Upload</button>
    </div>

    <div id='risk'>
        <label>Risk tolerance <input id='risk_tolerance' type='number' step='0.01'></label>
        <label>Max allocation <input id='max_allocation' type='number' step='0.01'></label>
        <label>Risk multiplier <input id='risk_multiplier' type='number' step='0.01'></label>
        <button id='save_risk'>Save</button>
    </div>

    <div id='discovery'>
        <label>Discovery
            <select id='discovery_select'>
                <option value='websocket'>websocket</option>
                <option value='mempool'>mempool</option>
                <option value='onchain'>onchain</option>
                <option value='pools'>pools</option>
                <option value='file'>file</option>
            </select>
        </label>
        <button id='save_discovery'>Save</button>
    </div>

    <div id='roi'></div>

    <table id='balances'>
        <thead><tr><th>Token</th><th>Amount</th><th>ROI</th></tr></thead>
        <tbody></tbody>
    </table>

    <h3>Recent Trades</h3>
    <table id='trades'>
        <thead><tr><th>Token</th><th>Side</th><th>Amount</th><th>Price</th><th>Time</th></tr></thead>
        <tbody></tbody>
    </table>

    <script>
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

    function loadKeypairs() {
        fetch('/keypairs').then(r => r.json()).then(data => {
            const sel = document.getElementById('keypair_select');
            sel.innerHTML = '';
            data.keypairs.forEach(n => {
                const opt = document.createElement('option');
                opt.value = n; opt.textContent = n; sel.appendChild(opt);
            });
            sel.value = data.active || '';
        });
    }

    function loadConfigs() {
        fetch('/configs').then(r => r.json()).then(data => {
            const sel = document.getElementById('config_select');
            sel.innerHTML = '';
            data.configs.forEach(n => {
                const opt = document.createElement('option');
                opt.value = n; opt.textContent = n; sel.appendChild(opt);
            });
            sel.value = data.active || '';
        });
    }

    function loadDiscovery() {
        fetch('/discovery').then(r => r.json()).then(data => {
            document.getElementById('discovery_select').value = data.method;
        });
    }
    document.getElementById('save_discovery').onclick = function() {
        const method = document.getElementById('discovery_select').value;
        fetch('/discovery', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({method})});
    };

    document.getElementById('keypair_select').onchange = function() {
        fetch('/keypairs/select', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name:this.value})});
    };
    document.getElementById('upload_keypair').onclick = function() {
        const file = document.getElementById('keypair_file').files[0];
        const name = document.getElementById('keypair_name').value;
        const fd = new FormData(); fd.append('file', file); fd.append('name', name);
        fetch('/keypairs/upload', {method:'POST', body:fd}).then(() => loadKeypairs());
    };

    document.getElementById('config_select').onchange = function() {
        fetch('/configs/select', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name:this.value})});
    };
    document.getElementById('upload_config').onclick = function() {
        const file = document.getElementById('config_file').files[0];
        const name = document.getElementById('config_name').value;
        const fd = new FormData(); fd.append('file', file); fd.append('name', name);
        fetch('/configs/upload', {method:'POST', body:fd}).then(() => loadConfigs());
    };

    function loadPositions() {
        fetch('/positions').then(r=>r.json()).then(data=>{
            const tbody=document.querySelector('#balances tbody');
            tbody.innerHTML='';
            Object.entries(data).forEach(([t,info])=>{
                const row=document.createElement('tr');
                row.innerHTML=`<td>${t}</td><td>${info.amount}</td><td>${info.roi.toFixed(4)}</td>`;
                tbody.appendChild(row);
            });
        });
    }

    function loadTrades() {
        fetch('/trades').then(r=>r.json()).then(data=>{
            const body=document.querySelector('#trades tbody');
            body.innerHTML='';
            data.forEach(tr=>{
                const row=document.createElement('tr');
                row.innerHTML=`<td>${tr.token}</td><td>${tr.direction}</td><td>${tr.amount}</td><td>${tr.price}</td><td>${tr.timestamp}</td>`;
                body.appendChild(row);
            });
        });
    }

    function loadRoi() {
        fetch('/roi').then(r=>r.json()).then(data=>{
            document.getElementById('roi').textContent = 'ROI: ' + data.roi.toFixed(4);
        });
    }

    loadRisk();
    loadKeypairs();
    loadConfigs();
    loadDiscovery();
    loadPositions();
    loadTrades();
    loadRoi();
    setInterval(loadPositions, 10000);
    setInterval(loadTrades, 10000);
    setInterval(loadRoi, 10000);

    </script>
</body>
</html>
"""

@app.route("/")
def index() -> str:
    return HTML_PAGE

if __name__ == "__main__":
    app.run()
