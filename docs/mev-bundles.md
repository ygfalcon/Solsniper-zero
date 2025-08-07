# MEV Bundles

When `use_mev_bundles` is enabled (the default), swaps are submitted
through the [Jito block-engine](https://jito.network/). The same
credentials can also be used to subscribe to Jito's searcher websocket
for real‑time pending transactions. Provide the block-engine and
websocket endpoints and authentication token:

```bash
export JITO_RPC_URL=https://block-engine.example.com
export JITO_AUTH=your_token
export JITO_WS_URL=wss://searcher.example.com
export JITO_WS_AUTH=your_token
```

The sniper and sandwich agents automatically pass these credentials to
`MEVExecutor` and will read pending transactions from the Jito stream
when both variables are set. A warning is logged if either variable is
missing while MEV bundles are enabled.
