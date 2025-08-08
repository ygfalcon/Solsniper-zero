document.getElementById('start').onclick = function() {
    fetch('/start_all', {method: 'POST'}).then(r => r.json()).then(console.log);
};
document.getElementById('stop').onclick = function() {
    fetch('/stop_all', {method: 'POST'}).then(r => r.json()).then(console.log);
};

const roiChart = new Chart(document.getElementById('roi_chart'), {
    type: 'line',
    data: {labels: [], datasets: [{label: 'ROI', data: [], borderColor: [], backgroundColor: []}]},
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

const varChart = new Chart(document.getElementById('var_chart'), {
    type: 'line',
    data: {labels: [], datasets: [{label:'VaR', data:[]}]},
    options: {scales:{y:{beginAtZero:true}}}
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
        document.getElementById('active_keypair').textContent = data.active || '';
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

function loadStrategies() {
    fetch('/strategies').then(r => r.json()).then(data => {
        const div = document.getElementById('strategy_controls');
        div.innerHTML = '';
        (data.available || []).forEach(name => {
            const label = document.createElement('label');
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.dataset.strategy = name;
            if(!data.active || data.active.includes(name)) cb.checked = true;
            label.appendChild(cb);
            label.appendChild(document.createTextNode(name));
            div.appendChild(label);
        });
    });
}

function loadConfig() {
    fetch('/configs').then(r => r.json()).then(data => {
        document.getElementById('active_config').textContent = data.active || '';
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

document.getElementById('save_strategies').onclick = function() {
    const names = [];
    document.querySelectorAll('#strategy_controls input').forEach(inp => {
        if(inp.checked) names.push(inp.dataset.strategy);
    });
    fetch('/strategies', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify(names)
    });
};

function refreshData() {
    fetch('/status').then(r => r.json()).then(data => {
        document.getElementById('status_info').textContent = JSON.stringify(data, null, 2);
    });
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
        const color = data.roi >= 0 ? 'green' : 'red';
        roiChart.data.labels.push('');
        roiChart.data.datasets[0].data.push(data.roi);
        roiChart.data.datasets[0].borderColor.push(color);
        roiChart.data.datasets[0].backgroundColor.push(color);
        if(roiChart.data.labels.length>50){
            roiChart.data.labels.shift();
            roiChart.data.datasets[0].data.shift();
            roiChart.data.datasets[0].borderColor.shift();
            roiChart.data.datasets[0].backgroundColor.shift();
        }
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
    fetch('/vars').then(r => r.json()).then(data => {
        document.getElementById('var_values').textContent = JSON.stringify(data.slice(-10), null, 2);
        varChart.data.labels = data.map(()=>'');
        varChart.data.datasets[0].data = data.map(v=>v.value);
        varChart.update();
    });
    fetch('/exposure').then(r => r.json()).then(data => {
        document.getElementById('exposure').textContent = JSON.stringify(data, null, 2);
    });
    fetch('/sharpe').then(r => r.json()).then(data => {
        document.getElementById('sharpe_val').textContent = data.sharpe.toFixed(4);
    });
    fetch('/rl/status').then(r => r.json()).then(data => {
        document.getElementById('rl_status').textContent = JSON.stringify(data);
    });
    loadWeights();
    loadStrategies();
}

loadKeypairs();
loadConfig();
loadRisk();
loadWeights();
loadStrategies();
refreshData();
setInterval(function(){
    refreshData();
    loadConfig();
    loadKeypairs();
    loadStrategies();
}, 5000);
try {
    const rlSock = new WebSocket('ws://' + window.location.hostname + ':8767');
    rlSock.onmessage = function(ev) {
        try {
            const data = JSON.parse(ev.data);
            if('loss' in data && 'reward' in data) {
                document.getElementById('rl_status').textContent = JSON.stringify(data);
            }
        } catch(e) {}
    };
} catch(e) {}
