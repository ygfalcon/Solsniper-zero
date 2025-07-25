use std::{collections::HashMap, fs::OpenOptions, path::Path, sync::Arc};

use tokio::sync::Mutex;

use std::time::{Duration, Instant};

use anyhow::Result;
use futures_util::StreamExt;
use memmap2::{MmapMut, MmapOptions};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::{io::{AsyncReadExt, AsyncWriteExt}, net::UnixListener};
use tokio_tungstenite::connect_async;
use solana_client::nonblocking::rpc_client::RpcClient;

mod route;
use solana_sdk::{
    commitment_config::CommitmentLevel,
    message::VersionedMessage,
    signature::{read_keypair_file, Keypair, Signer},
    system_instruction,
    transaction::{Transaction, VersionedTransaction},
};
use base64::{engine::general_purpose::STANDARD, Engine};
use bincode::{deserialize, serialize};
use solana_client::rpc_config::RpcSendTransactionConfig;
use chrono::Utc;

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
struct TokenInfo {
    bids: f64,
    asks: f64,
    tx_rate: f64,
}

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
struct TokenAgg {
    dex: HashMap<String, TokenInfo>,
    bids: f64,
    asks: f64,
    tx_rate: f64,
    ts: i64,
}

type DexMap = Arc<Mutex<HashMap<String, HashMap<String, TokenInfo>>>>;
type MempoolMap = Arc<Mutex<HashMap<String, f64>>>;

type SharedMmap = Arc<Mutex<MmapMut>>;

#[derive(Default, Clone)]
struct AutoExecEntry {
    threshold: f64,
    txs: Vec<String>,
}

type AutoExecMap = Arc<Mutex<HashMap<String, AutoExecEntry>>>;

struct ExecContext {
    client: RpcClient,
    keypair: Keypair,
    blockhash: Arc<Mutex<(solana_sdk::hash::Hash, Instant)>>,
}

impl ExecContext {
    async fn new(url: &str, keypair: Keypair) -> Self {
        let client = RpcClient::new(url.to_string());
        let bh = client.get_latest_blockhash().await.unwrap();
        Self {
            client,
            keypair,
            blockhash: Arc::new(Mutex::new((bh, Instant::now()))),
        }
    }

    async fn latest_blockhash(&self) -> solana_sdk::hash::Hash {
        let mut guard = self.blockhash.lock().await;
        if guard.1.elapsed() > Duration::from_secs(30) {
            if let Ok(new) = self.client.get_latest_blockhash().await {
                *guard = (new, Instant::now());
            }
        }
        guard.0
    }

    async fn send_dummy_tx(&self) -> Result<String> {
        let bh = self.latest_blockhash().await;
        let ix = system_instruction::transfer(&self.keypair.pubkey(), &self.keypair.pubkey(), 0);
        let tx = Transaction::new_signed_with_payer(
            &[ix],
            Some(&self.keypair.pubkey()),
            &[&self.keypair],
            bh,
        );
        let sig = self.client.send_transaction(&tx).await?;
        Ok(sig.to_string())
    }

    async fn send_raw_tx(&self, tx_b64: &str) -> Result<String> {
        let data = STANDARD.decode(tx_b64)?;
        let tx: VersionedTransaction = deserialize(&data)?;
        let config = RpcSendTransactionConfig {
            skip_preflight: true,
            preflight_commitment: Some(CommitmentLevel::Processed),
            ..RpcSendTransactionConfig::default()
        };
        let sig = self
            .client
            .send_transaction_with_config(&tx, config)
            .await?;
        Ok(sig.to_string())
    }

    async fn send_raw_tx_priority(
        &self,
        tx_b64: &str,
        priority: Option<Vec<String>>,
    ) -> Result<String> {
        let data = STANDARD.decode(tx_b64)?;
        let tx: VersionedTransaction = deserialize(&data)?;
        let config = RpcSendTransactionConfig {
            skip_preflight: true,
            preflight_commitment: Some(CommitmentLevel::Processed),
            ..RpcSendTransactionConfig::default()
        };
        if let Some(urls) = priority {
            for url in urls {
                let client = RpcClient::new(url);
                if let Ok(sig) = client
                    .send_transaction_with_config(&tx, config.clone())
                    .await
                {
                    return Ok(sig.to_string());
                }
            }
        }
        let sig = self
            .client
            .send_transaction_with_config(&tx, config)
            .await?;
        Ok(sig.to_string())
    }

    async fn submit_signed_tx(&self, tx_b64: &str) -> Result<String> {
        self.send_raw_tx(tx_b64).await
    }

    async fn sign_template(
        &self,
        msg_b64: &str,
        priority_fee: Option<u64>,
    ) -> Result<String> {
        use solana_sdk::{
            compute_budget::ComputeBudgetInstruction,
            message::MessageHeader,
        };

        let mut msg: VersionedMessage = deserialize(&STANDARD.decode(msg_b64)?)?;

        if let Some(fee) = priority_fee {
            fn insert_budget(
                header: &mut MessageHeader,
                keys: &mut Vec<solana_sdk::pubkey::Pubkey>,
                ixs: &mut Vec<solana_sdk::instruction::CompiledInstruction>,
                fee: u64,
            ) {
                use solana_sdk::{compute_budget, message::v0::LoadedAddresses};
                use solana_program::message::AccountKeys;
                let pid = compute_budget::id();
                if keys.iter().position(|k| *k == pid).is_none() {
                    keys.push(pid);
                    header.num_readonly_unsigned_accounts = header
                        .num_readonly_unsigned_accounts
                        .saturating_add(1);
                };
                let accounts = AccountKeys::new(keys, None::<&LoadedAddresses>);
                let mut add_ix = |ix: solana_sdk::instruction::Instruction| {
                    let mut c = accounts.compile_instructions(&[ix]);
                    if let Some(ci) = c.pop() {
                        ixs.insert(0, ci);
                    }
                };
                add_ix(ComputeBudgetInstruction::set_compute_unit_price(fee));
                add_ix(ComputeBudgetInstruction::set_compute_unit_limit(1_400_000));
            }

            match &mut msg {
                VersionedMessage::Legacy(m) => {
                    insert_budget(&mut m.header, &mut m.account_keys, &mut m.instructions, fee);
                }
                VersionedMessage::V0(m) => {
                    insert_budget(&mut m.header, &mut m.account_keys, &mut m.instructions, fee);
                }
            }
        }

        let bh = self.latest_blockhash().await;
        match &mut msg {
            VersionedMessage::Legacy(m) => m.recent_blockhash = bh,
            VersionedMessage::V0(m) => m.recent_blockhash = bh,
        }
        let tx = VersionedTransaction::try_new(msg, &[&self.keypair])?;
        Ok(STANDARD.encode(serialize(&tx)?))
    }

    async fn send_batch(&self, txs: &[String]) -> Result<Vec<String>> {
        let mut sigs = Vec::new();
        for tx in txs {
            sigs.push(self.send_raw_tx(tx).await?);
        }
        Ok(sigs)
    }
}

async fn update_mmap(dex_map: &DexMap, mem: &MempoolMap, mmap: &SharedMmap) -> Result<()> {
    let dexes = dex_map.lock().await;
    let mem = mem.lock().await;
    let mut agg: HashMap<String, TokenAgg> = HashMap::new();
    for (dex_name, dex) in dexes.iter() {
        for (tok, info) in dex {
            let entry = agg.entry(tok.clone()).or_default();
            entry.dex.insert(dex_name.clone(), info.clone());
            entry.bids += info.bids;
            entry.asks += info.asks;
            if entry.ts == 0 {
                entry.ts = Utc::now().timestamp_millis();
            }
        }
    }
    for (tok, rate) in mem.iter() {
        let entry = agg.entry(tok.clone()).or_default();
        entry.tx_rate = *rate;
        entry.ts = Utc::now().timestamp_millis();
    }
    let mut json = serde_json::to_vec(&agg)?;
    let mut mmap = mmap.lock().await;
    if json.len() > mmap.len() {
        json.truncate(mmap.len());
    }
    mmap[..json.len()].copy_from_slice(&json);
    for i in json.len()..mmap.len() {
        mmap[i] = 0;
    }
    mmap.flush()?;
    Ok(())
}

async fn connect_feed(dex: &str, url: &str, dex_map: DexMap, mem: MempoolMap, mmap: SharedMmap) -> Result<()> {
    let (ws, _) = connect_async(url).await?;
    let (_write, mut read) = ws.split();
    while let Some(Ok(msg)) = read.next().await {
        if !msg.is_text() {
            continue;
        }
        if let Ok(val) = serde_json::from_str::<Value>(&msg.to_string()) {
            if let (Some(token), Some(bids), Some(asks)) = (
                val.get("token"),
                val.get("bids"),
                val.get("asks"),
            ) {
                let token = token.as_str().unwrap_or("").to_string();
                let bids = bids.as_f64().unwrap_or(0.0);
                let asks = asks.as_f64().unwrap_or(0.0);
                let mut dexes = dex_map.lock().await;
                let entry = dexes.entry(dex.to_string()).or_default();
                entry.insert(token.clone(), TokenInfo { bids, asks, tx_rate: 0.0 });
                drop(dexes);
                let _ = update_mmap(&dex_map, &mem, &mmap).await;
            }
        }
    }
    Ok(())
}

async fn connect_price_feed(dex: &str, url: &str, dex_map: DexMap, mem: MempoolMap, mmap: SharedMmap) -> Result<()> {
    let (ws, _) = connect_async(url).await?;
    let (_write, mut read) = ws.split();
    while let Some(Ok(msg)) = read.next().await {
        if !msg.is_text() {
            continue;
        }
        if let Ok(val) = serde_json::from_str::<Value>(&msg.to_string()) {
            if let (Some(token), Some(price)) = (val.get("token"), val.get("price")) {
                let token = token.as_str().unwrap_or("").to_string();
                let price = price.as_f64().unwrap_or(0.0);
                let mut dexes = dex_map.lock().await;
                let entry = dexes.entry(dex.to_string()).or_default();
                entry.insert(token.clone(), TokenInfo { bids: price, asks: price, tx_rate: 0.0 });
                drop(dexes);
                let _ = update_mmap(&dex_map, &mem, &mmap).await;
            }
        }
    }
    Ok(())
}

async fn connect_mempool(url: &str, mem: MempoolMap, dex_map: DexMap, mmap: SharedMmap) -> Result<()> {
    let (ws, _) = connect_async(url).await?;
    let (_write, mut read) = ws.split();
    while let Some(Ok(msg)) = read.next().await {
        if !msg.is_text() {
            continue;
        }
        if let Ok(val) = serde_json::from_str::<Value>(&msg.to_string()) {
            if let (Some(token), Some(rate)) = (val.get("token"), val.get("tx_rate")) {
                let token = token.as_str().unwrap_or("").to_string();
                let rate = rate.as_f64().unwrap_or(0.0);
                let mut mem_lock = mem.lock().await;
                mem_lock.insert(token, rate);
                drop(mem_lock);
                let _ = update_mmap(&dex_map, &mem, &mmap).await;
            }
        }
    }
    Ok(())
}

async fn ipc_server(socket: &Path, dex_map: DexMap, mem: MempoolMap, exec: Arc<ExecContext>) -> Result<()> {
    if socket.exists() {
        let _ = std::fs::remove_file(socket);
    }
    let listener = UnixListener::bind(socket)?;
    let auto_map: AutoExecMap = Arc::new(Mutex::new(HashMap::new()));

    {
        let d = dex_map.clone();
        let m = mem.clone();
        let exec = exec.clone();
        let a = auto_map.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(200)).await;
                let entries = { a.lock().await.clone() };
                if entries.is_empty() {
                    continue;
                }
                let rates = { m.lock().await.clone() };
                let dexes = { d.lock().await.clone() };
                let mut remove = Vec::new();
                for (tok, cfg) in entries.iter() {
                    let rate = rates.get(tok).copied().unwrap_or(0.0);
                    let mut liq = 0.0;
                    for dex in dexes.values() {
                        if let Some(info) = dex.get(tok) {
                            liq += info.bids + info.asks;
                        }
                    }
                    if rate >= cfg.threshold || liq >= cfg.threshold {
                        for tx in &cfg.txs {
                            let _ = exec.submit_signed_tx(tx).await;
                        }
                        remove.push(tok.clone());
                    }
                }
                if !remove.is_empty() {
                    let mut lock = a.lock().await;
                    for t in remove { lock.remove(&t); }
                }
            }
        });
    }
    loop {
        let (mut stream, _) = listener.accept().await?;
        let dex_map = dex_map.clone();
        let mem = mem.clone();
        let exec = exec.clone();
        let auto_map = auto_map.clone();
        tokio::spawn(async move {
            let mut buf = Vec::new();
            if stream.read_to_end(&mut buf).await.is_ok() {
                if let Ok(val) = serde_json::from_slice::<Value>(&buf) {
                    if val.get("cmd") == Some(&Value::String("snapshot".into())) {
                        if let Some(token) = val.get("token").and_then(|v| v.as_str()) {
                            let dexes = dex_map.lock().await;
                            let mem = mem.lock().await;
                            let mut entry = TokenAgg::default();
                            for (dex_name, dex) in dexes.iter() {
                                if let Some(info) = dex.get(token) {
                                    entry.bids += info.bids;
                                    entry.asks += info.asks;
                                    entry.dex.insert(dex_name.clone(), info.clone());
                                }
                            }
                            if let Some(rate) = mem.get(token) {
                                entry.tx_rate = *rate;
                            }
                            let _ = stream.write_all(serde_json::to_string(&entry).unwrap().as_bytes()).await;
                        }
                    } else if val.get("cmd") == Some(&Value::String("signed_tx".into())) {
                        if let Some(tx) = val.get("tx").and_then(|v| v.as_str()) {
                            match exec.submit_signed_tx(tx).await {
                                Ok(sig) => {
                                    let _ = stream
                                        .write_all(format!("{{\"signature\":\"{}\"}}", sig).as_bytes())
                                        .await;
                                }
                                Err(e) => {
                                    let _ = stream
                                        .write_all(format!("{{\"error\":\"{}\"}}", e).as_bytes())
                                        .await;
                                }
                            }
                        }
                    } else if val.get("cmd") == Some(&Value::String("prepare".into())) {
                        if let Some(msg) = val.get("msg").and_then(|v| v.as_str()) {
                            let pf = val.get("priority_fee").and_then(|v| v.as_u64());
                            match exec.sign_template(msg, pf).await {
                                Ok(tx) => {
                                    let _ = stream
                                        .write_all(format!("{{\"tx\":\"{}\"}}", tx).as_bytes())
                                        .await;
                                }
                                Err(e) => {
                                    let _ = stream
                                        .write_all(format!("{{\"error\":\"{}\"}}", e).as_bytes())
                                        .await;
                                }
                            }
                        }
                    } else if val.get("cmd") == Some(&Value::String("batch".into())) {
                        if let Some(arr) = val.get("txs").and_then(|v| v.as_array()) {
                            let txs: Vec<String> = arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect();
                            match exec.send_batch(&txs).await {
                                Ok(sigs) => {
                                    let _ = stream
                                        .write_all(serde_json::to_string(&sigs).unwrap().as_bytes())
                                        .await;
                                }
                                Err(e) => {
                                    let _ = stream
                                        .write_all(format!("{{\"error\":\"{}\"}}", e).as_bytes())
                                        .await;
                                }
                            }
                        }
                    } else if val.get("cmd") == Some(&Value::String("raw_tx".into())) {
                        if let Some(tx) = val.get("tx").and_then(|v| v.as_str()) {
                            let pri = val
                                .get("priority_rpc")
                                .and_then(|v| v.as_array())
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                        .collect::<Vec<String>>()
                                });
                            match exec.send_raw_tx_priority(tx, pri).await {
                                Ok(sig) => {
                                    let _ = stream
                                        .write_all(format!("{{\"signature\":\"{}\"}}", sig).as_bytes())
                                        .await;
                                }
                                Err(e) => {
                                    let _ = stream
                                        .write_all(format!("{{\"error\":\"{}\"}}", e).as_bytes())
                                        .await;
                                }
                            }
                        }
                    } else if val.get("cmd") == Some(&Value::String("submit".into())) {
                        if let Some(tx) = val.get("tx").and_then(|v| v.as_str()) {
                            match exec.send_raw_tx(tx).await {
                                Ok(sig) => {
                                    let _ = stream
                                        .write_all(format!("{{\"signature\":\"{}\"}}", sig).as_bytes())
                                        .await;
                                }
                                Err(e) => {
                                    let _ = stream
                                        .write_all(format!("{{\"error\":\"{}\"}}", e).as_bytes())
                                        .await;
                                }
                            }
                        }
                    } else if val.get("cmd") == Some(&Value::String("order".into())) {
                        match exec.send_dummy_tx().await {
                            Ok(sig) => {
                                let _ = stream
                                    .write_all(format!("{{\"signature\":\"{}\"}}", sig).as_bytes())
                                    .await;
                            }
                            Err(_) => {
                                let _ = stream.write_all(b"{\"ok\":false}").await;
                            }
                        }
                    } else if val.get("cmd") == Some(&Value::String("auto_exec".into())) {
                        if let (Some(token), Some(th), Some(arr)) = (
                            val.get("token").and_then(|v| v.as_str()),
                            val.get("threshold").and_then(|v| v.as_f64()),
                            val.get("txs").and_then(|v| v.as_array()),
                        ) {
                            let txs: Vec<String> = arr
                                .iter()
                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                .collect();
                            auto_map.lock().await.insert(
                                token.to_string(),
                                AutoExecEntry { threshold: th, txs },
                            );
                            let _ = stream.write_all(b"{\"ok\":true}").await;
                        }
                    } else if val.get("cmd") == Some(&Value::String("route".into())) {
                        if let (Some(token), Some(amount)) = (
                            val.get("token").and_then(|v| v.as_str()),
                            val.get("amount").and_then(|v| v.as_f64()),
                        ) {
                            let dexes = dex_map.lock().await;
                            if let Some(res) = route::best_route(&dexes, token, amount, 4) {
                                let resp = serde_json::json!({
                                    "path": res.path,
                                    "profit": res.profit,
                                    "slippage": res.slippage
                                });
                                let _ = stream.write_all(resp.to_string().as_bytes()).await;
                            } else {
                                let _ = stream.write_all(b"{}\n").await;
                            }
                        }
                    }
                }
            }
            let _ = stream.shutdown().await;
        });
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut serum = None;
    let mut raydium = None;
    let mut orca = None;
    let mut jupiter = None;
    let mut phoenix = None;
    let mut meteora = None;
    let mut mempool = None;
    let mut rpc = std::env::var("SOLANA_RPC_URL").unwrap_or_default();
    let mut keypair_path = std::env::var("SOLANA_KEYPAIR").unwrap_or_default();
    for w in args.windows(2) {
        match w[0].as_str() {
            "--serum" => serum = Some(w[1].clone()),
            "--raydium" => raydium = Some(w[1].clone()),
            "--rpc" => rpc = w[1].clone(),
            "--keypair" => keypair_path = w[1].clone(),
            "--mempool" => mempool = Some(w[1].clone()),
            "--orca" => orca = Some(w[1].clone()),
            "--jupiter" => jupiter = Some(w[1].clone()),
            "--phoenix" => phoenix = Some(w[1].clone()),
            "--meteora" => meteora = Some(w[1].clone()),
            _ => {}
        }
    }
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open("/tmp/depth_service.mmap")?;
    file.set_len(65536)?;
    let mmap = unsafe { MmapOptions::new().map_mut(&file)? };
    let mmap = Arc::new(Mutex::new(mmap));
    let dex_map: DexMap = Arc::new(Mutex::new(HashMap::new()));
    let mem_map: MempoolMap = Arc::new(Mutex::new(HashMap::new()));

    if let Some(url) = serum {
        let d = dex_map.clone();
        let m = mem_map.clone();
        let mm = mmap.clone();
        tokio::spawn(async move {
            let _ = connect_feed("serum", &url, d, m, mm).await;
        });
    }
    if let Some(url) = raydium {
        let d = dex_map.clone();
        let m = mem_map.clone();
        let mm = mmap.clone();
        tokio::spawn(async move {
            let _ = connect_price_feed("raydium", &url, d, m, mm).await;
        });
    }
    if let Some(url) = orca {
        let d = dex_map.clone();
        let m = mem_map.clone();
        let mm = mmap.clone();
        tokio::spawn(async move {
            let _ = connect_price_feed("orca", &url, d, m, mm).await;
        });
    }
    if let Some(url) = jupiter {
        let d = dex_map.clone();
        let m = mem_map.clone();
        let mm = mmap.clone();
        tokio::spawn(async move {
            let _ = connect_price_feed("jupiter", &url, d, m, mm).await;
        });
    }
    if let Some(url) = phoenix {
        let d = dex_map.clone();
        let m = mem_map.clone();
        let mm = mmap.clone();
        tokio::spawn(async move {
            let _ = connect_price_feed("phoenix", &url, d, m, mm).await;
        });
    }
    if let Some(url) = meteora {
        let d = dex_map.clone();
        let m = mem_map.clone();
        let mm = mmap.clone();
        tokio::spawn(async move {
            let _ = connect_price_feed("meteora", &url, d, m, mm).await;
        });
    }
    if let Some(url) = mempool {
        let d = dex_map.clone();
        let m = mem_map.clone();
        let mm = mmap.clone();
        tokio::spawn(async move {
            let _ = connect_mempool(&url, m, d, mm).await;
        });
    }

    let kp = if !keypair_path.is_empty() {
        read_keypair_file(&keypair_path).unwrap_or_else(|_| Keypair::new())
    } else {
        Keypair::new()
    };
    if rpc.is_empty() {
        rpc = "https://api.mainnet-beta.solana.com".to_string();
    }
    let exec = Arc::new(ExecContext::new(&rpc, kp).await);

    ipc_server(Path::new("/tmp/depth_service.sock"), dex_map, mem_map, exec).await?;
    Ok(())
}
