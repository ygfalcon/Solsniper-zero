document.addEventListener('DOMContentLoaded', () => {
  try {
    const logEl = document.getElementById('startup-log-content');
    const ws = new WebSocket('ws://' + window.location.hostname + ':8766/ws');
    ws.onmessage = ev => {
      try {
        const data = JSON.parse(ev.data);
        if (data.topic === 'log' && logEl) {
          logEl.textContent += data.payload + '\n';
        }
      } catch (e) {
        console.error('Failed to parse message', e);
      }
    };
  } catch (e) {
    console.error('WebSocket connection failed', e);
  }
});

