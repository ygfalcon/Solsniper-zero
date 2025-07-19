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

def trading_loop() -> None:
    memory = Memory("sqlite:///memory.db")
    portfolio = Portfolio()
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


@app.route("/keypairs", methods=["GET"])
def list_keypairs() -> dict:
    return jsonify({"keypairs": wallet.list_keypairs(), "active": wallet.get_active_keypair_name()})


@app.route("/keypairs", methods=["POST"])
def add_keypair() -> dict:
    data = request.get_json(force=True)
    name = data.get("name")
    keypair_data = data.get("keypair")
    if isinstance(keypair_data, str):
        keypair_data = json.loads(keypair_data)
    wallet.save_keypair(name, keypair_data)
    return jsonify({"status": "saved"})


@app.route("/keypairs/select", methods=["POST"])
def select_keypair() -> dict:
    data = request.get_json(force=True)
    name = data.get("name")
    wallet.select_keypair(name)
    return jsonify({"status": "selected"})

HTML_PAGE = """
<!doctype html>
<html>
<head>
    <title>SolHunter UI</title>
</head>
<body>
    <button id='start'>Start</button>
    <button id='stop'>Stop</button>
    <form id='add-form'>
        <input name='name' placeholder='Name'>
        <textarea name='keypair' placeholder='Keypair JSON'></textarea>
        <button type='submit'>Add</button>
    </form>
    <select id='keypair-select'></select>
    <script>
    document.getElementById('start').onclick = function() {
        fetch('/start', {method: 'POST'}).then(r => r.json()).then(console.log);
    };
    document.getElementById('stop').onclick = function() {
        fetch('/stop', {method: 'POST'}).then(r => r.json()).then(console.log);
    };
    function refreshKeypairs() {
        fetch('/keypairs').then(r => r.json()).then(data => {
            const sel = document.getElementById('keypair-select');
            sel.innerHTML = '';
            data.keypairs.forEach(n => {
                const opt = document.createElement('option');
                opt.value = n;
                opt.textContent = n;
                if (n === data.active) opt.selected = true;
                sel.appendChild(opt);
            });
        });
    }
    refreshKeypairs();
    document.getElementById('add-form').onsubmit = function(e) {
        e.preventDefault();
        const name = this.name.value;
        const keypair = this.keypair.value;
        fetch('/keypairs', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({name, keypair})}).then(refreshKeypairs);
    };
    document.getElementById('keypair-select').onchange = function() {
        fetch('/keypairs/select', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({name: this.value})});
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
