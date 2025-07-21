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
use solana_sdk::{
    signature::{read_keypair_file, Keypair, Signer},
    system_instruction,
    transaction::Transaction,
};

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
struct Depth {
    bids: f64,
    asks: f64,
}

type DepthMap = Arc<Mutex<HashMap<String, Depth>>>;

type SharedMmap = Arc<Mutex<MmapMut>>;

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
}

async fn update_mmap(map: &DepthMap, mmap: &SharedMmap) -> Result<()> {
    let map = map.lock().await.clone();
    let mut json = serde_json::to_vec(&map)?;
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

async fn connect_feed(url: &str, map: DepthMap, mmap: SharedMmap) -> Result<()> {
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
                let mut map_lock = map.lock().await;
                map_lock.insert(token, Depth { bids, asks });
                drop(map_lock);
                let _ = update_mmap(&map, &mmap).await;
            }
        }
    }
    Ok(())
}

async fn ipc_server(socket: &Path, map: DepthMap, exec: Arc<ExecContext>) -> Result<()> {
    if socket.exists() {
        let _ = std::fs::remove_file(socket);
    }
    let listener = UnixListener::bind(socket)?;
    loop {
        let (mut stream, _) = listener.accept().await?;
        let map = map.clone();
        let exec = exec.clone();
        tokio::spawn(async move {
            let mut buf = Vec::new();
            if stream.read_to_end(&mut buf).await.is_ok() {
                if let Ok(val) = serde_json::from_slice::<Value>(&buf) {
                    if val.get("cmd") == Some(&Value::String("snapshot".into())) {
                        if let Some(token) = val.get("token").and_then(|v| v.as_str()) {
                            let data = map.lock().await;
                            if let Some(d) = data.get(token) {
                                let _ = stream
                                    .write_all(serde_json::to_string(d).unwrap().as_bytes())
                                    .await;
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
    let mut rpc = std::env::var("SOLANA_RPC_URL").unwrap_or_default();
    let mut keypair_path = std::env::var("SOLANA_KEYPAIR").unwrap_or_default();
    for w in args.windows(2) {
        match w[0].as_str() {
            "--serum" => serum = Some(w[1].clone()),
            "--raydium" => raydium = Some(w[1].clone()),
            "--rpc" => rpc = w[1].clone(),
            "--keypair" => keypair_path = w[1].clone(),
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
    let map: DepthMap = Arc::new(Mutex::new(HashMap::new()));

    let m1 = map.clone();
    let mm1 = mmap.clone();
    if let Some(url) = serum {
        tokio::spawn(async move {
            let _ = connect_feed(&url, m1, mm1).await;
        });
    }
    let m2 = map.clone();
    let mm2 = mmap.clone();
    if let Some(url) = raydium {
        tokio::spawn(async move {
            let _ = connect_feed(&url, m2, mm2).await;
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

    ipc_server(Path::new("/tmp/depth_service.sock"), map, exec).await?;
    Ok(())
}
