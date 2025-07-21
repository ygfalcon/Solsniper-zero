use std::{collections::HashMap, fs::OpenOptions, path::Path, sync::{Arc, Mutex}};

use anyhow::Result;
use futures_util::StreamExt;
use memmap2::{MmapMut, MmapOptions};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::{io::{AsyncReadExt, AsyncWriteExt}, net::UnixListener};
use tokio_tungstenite::connect_async;

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
struct Depth {
    bids: f64,
    asks: f64,
}

type DepthMap = Arc<Mutex<HashMap<String, Depth>>>;

type SharedMmap = Arc<Mutex<MmapMut>>;

async fn update_mmap(map: &DepthMap, mmap: &SharedMmap) -> Result<()> {
    let map = map.lock().unwrap().clone();
    let mut json = serde_json::to_vec(&map)?;
    let mut mmap = mmap.lock().unwrap();
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
                let mut map_lock = map.lock().unwrap();
                map_lock.insert(token, Depth { bids, asks });
                drop(map_lock);
                let _ = update_mmap(&map, &mmap).await;
            }
        }
    }
    Ok(())
}

async fn ipc_server(socket: &Path, map: DepthMap) -> Result<()> {
    if socket.exists() {
        let _ = std::fs::remove_file(socket);
    }
    let listener = UnixListener::bind(socket)?;
    loop {
        let (mut stream, _) = listener.accept().await?;
        let map = map.clone();
        tokio::spawn(async move {
            let mut buf = Vec::new();
            if stream.read_to_end(&mut buf).await.is_ok() {
                if let Ok(val) = serde_json::from_slice::<Value>(&buf) {
                    if val.get("cmd") == Some(&Value::String("snapshot".into())) {
                        if let Some(token) = val.get("token").and_then(|v| v.as_str()) {
                            let data = map.lock().unwrap();
                            if let Some(d) = data.get(token) {
                                let _ = stream
                                    .write_all(serde_json::to_string(d).unwrap().as_bytes())
                                    .await;
                            }
                        }
                    } else if val.get("cmd") == Some(&Value::String("order".into())) {
                        // Placeholder: simply acknowledge
                        let _ = stream.write_all(b"{\"ok\":true}").await;
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
    for w in args.windows(2) {
        match w[0].as_str() {
            "--serum" => serum = Some(w[1].clone()),
            "--raydium" => raydium = Some(w[1].clone()),
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

    ipc_server(Path::new("/tmp/depth_service.sock"), map).await?;
    Ok(())
}
