import threading
import time
import os
import asyncio
import json
import logging
from collections import deque
from flask import Flask, jsonify, request
from pathlib import Path

from .config import load_config, apply_env_overrides, set_env_from_config
from . import config as config_module

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

if get_active_config_name() is None and _DEFAULT_PRESET.is_file():
    dest = Path(config_module.CONFIG_DIR) / _DEFAULT_PRESET.name
    if not dest.exists():
        dest.write_bytes(_DEFAULT_PRESET.read_bytes())
    select_config(dest.name)

cfg = apply_env_overrides(cfg)
set_env_from_config(cfg)

app = Flask(__name__)

# in-memory log storage for UI access
log_buffer: deque[str] = deque(maxlen=200)


class _BufferHandler(logging.Handler):
    """Logging handler that stores formatted log records in ``log_buffer``."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - simple
        log_buffer.append(self.format(record))


buffer_handler = _BufferHandler()
buffer_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logging.getLogger().addHandler(buffer_handler)

trading_thread = None
stop_event = threading.Event()
loop_delay = 60

# currently active portfolio and keypair used by the trading loop
current_portfolio: Portfolio | None = None
current_keypair = None
pnl_history: list[float] = []
token_pnl_history: dict[str, list[float]] = {}
allocation_history: dict[str, list[float]] = {}

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

    # auto-select the only available keypair if none is active
    if wallet.get_active_keypair_name() is None:
        keys = wallet.list_keypairs()
        if len(keys) == 1:
            wallet.select_keypair(keys[0])

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


@app.route("/weights", methods=["GET", "POST"])
def agent_weights() -> dict:
    """Get or update agent weighting factors."""
    if request.method == "POST":
        weights = request.get_json() or {}
        os.environ["AGENT_WEIGHTS"] = json.dumps(weights)
        return jsonify({"status": "ok"})

    env = os.getenv("AGENT_WEIGHTS")
    if not env:
        return jsonify({})
    try:
        return jsonify(json.loads(env))
    except Exception:
        try:
            import ast

            return jsonify(ast.literal_eval(env))
        except Exception:
            return jsonify({})


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


@app.route("/pnl")
def pnl() -> dict:
    pf = current_portfolio or Portfolio()
    tokens = list(pf.balances.keys())
    prices = fetch_token_prices(tokens)
    entry = sum(p.amount * p.entry_price for p in pf.balances.values())
    value = sum(p.amount * prices.get(tok, p.entry_price) for tok, p in pf.balances.items())
    pnl = value - entry
    pnl_history.append(pnl)
    return jsonify({"pnl": pnl, "history": pnl_history})


@app.route("/token_history")
def token_history() -> dict:
    pf = current_portfolio or Portfolio()
    tokens = list(pf.balances.keys())
    prices = fetch_token_prices(tokens)
    total = pf.total_value(prices)
    result: dict[str, dict[str, list[float]]] = {}
    for token, pos in pf.balances.items():
        price = prices.get(token, pos.entry_price)
        pnl = (price - pos.entry_price) * pos.amount
        token_pnl_history.setdefault(token, []).append(pnl)
        alloc = (pos.amount * price) / total if total else 0.0
        allocation_history.setdefault(token, []).append(alloc)
        result[token] = {
            "pnl_history": token_pnl_history[token],
            "allocation_history": allocation_history[token],
        }
    return jsonify(result)



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


@app.route("/logs")
def logs() -> dict:
    """Return recent log messages."""
    return jsonify({"logs": list(log_buffer)})


HTML_PAGE = """
<!doctype html>
<html>
<head>
    <title>SolHunter UI</title>
    <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
</head>
<body>
    <button id='start'>Start</button>
    <button id='stop'>Stop</button>
    <select id='keypair_select'></select>

    <h3>ROI: <span id='roi_value'>0</span></h3>
    <canvas id='roi_chart' width='400' height='100'></canvas>

    <h3>Positions</h3>
    <pre id='positions'></pre>

    <h3>Recent Trades</h3>
    <pre id='trades'></pre>
    <canvas id='trade_chart' width='400' height='100'></canvas>

    <h3>Agent Weights</h3>
    <div id='weights_controls'></div>
    <button id='save_weights'>Save Weights</button>
    <canvas id='weights_chart' width='400' height='100'></canvas>

    <h3>Token PnL</h3>
    <canvas id='pnl_chart' width='400' height='100'></canvas>

    <h3>Token Allocation</h3>
    <canvas id='allocation_chart' width='400' height='100'></canvas>

    <h3>Risk Parameters</h3>
    <label>Risk Tolerance <input id='risk_tolerance' type='number' step='0.01'></label>
    <label>Max Allocation <input id='max_allocation' type='number' step='0.01'></label>
    <label>Risk Multiplier <input id='risk_multiplier' type='number' step='0.01'></label>
    <button id='save_risk'>Save Risk</button>

    <script>
    document.getElementById('start').onclick = function() {
        fetch('/start', {method: 'POST'}).then(r => r.json()).then(console.log);
    };
    document.getElementById('stop').onclick = function() {
        fetch('/stop', {method: 'POST'}).then(r => r.json()).then(console.log);
    };

    const roiChart = new Chart(document.getElementById('roi_chart'), {
        type: 'line',
        data: {labels: [], datasets: [{label: 'ROI', data: []}]},
        options: {scales: {y: {beginAtZero: true}}}
    });

    const tradeChart = new Chart(document.getElementById('trade_chart'), {
        type: 'bar',
        data: {labels: ['buy', 'sell'], datasets: [{label:'Trades', data:[0,0]}]},
        options: {scales:{y:{beginAtZero:true}}}
    });

    const weightsChart = new Chart(document.getElementById('weights_chart'), {
        type: 'bar',
        data: {labels: [], datasets: [{label:'Weight', data: []}]},
        options: {scales:{y:{beginAtZero:true}}}
    });

    const pnlChart = new Chart(document.getElementById('pnl_chart'), {
        type: 'line',
        data: {labels: [], datasets: []},
        options: {scales:{y:{beginAtZero:true}}}
    });

    const allocationChart = new Chart(document.getElementById('allocation_chart'), {
        type: 'line',
        data: {labels: [], datasets: []},
        options: {scales:{y:{beginAtZero:true, max:1}}}
    });

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
    document.getElementById('keypair_select').onchange = function() {
        fetch('/keypairs/select', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name:this.value})});
    };

    function loadRisk() {
        fetch('/risk').then(r => r.json()).then(data => {
            document.getElementById('risk_tolerance').value = data.risk_tolerance;
            document.getElementById('max_allocation').value = data.max_allocation;
            document.getElementById('risk_multiplier').value = data.risk_multiplier;
        });
    }

    document.getElementById('save_risk').onclick = function() {
        fetch('/risk', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({
                risk_tolerance: parseFloat(document.getElementById('risk_tolerance').value),
                max_allocation: parseFloat(document.getElementById('max_allocation').value),
                risk_multiplier: parseFloat(document.getElementById('risk_multiplier').value)
            })
        });
    };

    function loadWeights() {
        fetch('/weights').then(r => r.json()).then(data => {
            const div = document.getElementById('weights_controls');
            div.innerHTML = '';
            const labels = [];
            const values = [];
            Object.entries(data).forEach(([name, val]) => {
                const label = document.createElement('label');
                label.textContent = name;
                const inp = document.createElement('input');
                inp.type = 'number';
                inp.step = '0.1';
                inp.value = val;
                inp.dataset.agent = name;
                label.appendChild(inp);
                div.appendChild(label);
                labels.push(name);
                values.push(val);
            });
            weightsChart.data.labels = labels;
            weightsChart.data.datasets[0].data = values;
            weightsChart.update();
        });
    }

    document.getElementById('save_weights').onclick = function() {
        const data = {};
        document.querySelectorAll('#weights_controls input').forEach(inp => {
            data[inp.dataset.agent] = parseFloat(inp.value);
        });
        fetch('/weights', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify(data)
        });
    };

    function refreshData() {
        fetch('/positions').then(r => r.json()).then(data => {
            document.getElementById('positions').textContent = JSON.stringify(data, null, 2);
        });
        fetch('/trades').then(r => r.json()).then(data => {
            document.getElementById('trades').textContent = JSON.stringify(data.slice(-10), null, 2);
            const buy = data.filter(t=>t.direction==='buy').length;
            const sell = data.filter(t=>t.direction==='sell').length;
            tradeChart.data.datasets[0].data = [buy, sell];
            tradeChart.update();
        });
        fetch('/roi').then(r => r.json()).then(data => {
            document.getElementById('roi_value').textContent = data.roi.toFixed(4);
            roiChart.data.labels.push('');
            roiChart.data.datasets[0].data.push(data.roi);
            if(roiChart.data.labels.length>50){roiChart.data.labels.shift();roiChart.data.datasets[0].data.shift();}
            roiChart.update();
        });
        fetch('/token_history').then(r => r.json()).then(data => {
            let maxLen = 0;
            Object.values(data).forEach(v => { if(v.pnl_history.length > maxLen) maxLen = v.pnl_history.length; });
            pnlChart.data.labels = Array.from({length:maxLen}, ()=>'');
            allocationChart.data.labels = pnlChart.data.labels;
            Object.entries(data).forEach(([tok, stats]) => {
                let ds = pnlChart.data.datasets.find(d => d.label===tok);
                if(!ds){ ds = {label:tok, data:[]}; pnlChart.data.datasets.push(ds); }
                ds.data = stats.pnl_history;
                let da = allocationChart.data.datasets.find(d => d.label===tok);
                if(!da){ da = {label:tok, data:[]}; allocationChart.data.datasets.push(da); }
                da.data = stats.allocation_history;
            });
            pnlChart.update();
            allocationChart.update();
        });
        loadWeights();
    }

    loadKeypairs();
    loadRisk();
    loadWeights();
    refreshData();
    setInterval(refreshData, 5000);
    </script>
</body>
</html>
"""

@app.route("/")
def index() -> str:
    return HTML_PAGE

if __name__ == "__main__":
    app.run()
