//! Path search utilities for depth_service.
//!
//! An in-memory LRU cache stores adjacency graphs keyed by `(token, max_hops)`.
//! Each graph represents price relationships between venues and is reused across
//! calls until invalidated. The cache holds up to 1024 entries. Call
//! [`invalidate_edges`] whenever depth updates introduce significant changes so
//! outdated graphs are dropped.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Mutex;

use lru::LruCache;
use once_cell::sync::Lazy;

type EdgeRates = (f64, f64); // (profit_rate, slip_rate)
type AdjMap = HashMap<String, HashMap<String, EdgeRates>>;

#[derive(Clone)]
struct CachedGraph {
    adjacency: AdjMap,
    max_diff: f64,
    venues: Vec<String>,
}

static EDGE_CACHE: Lazy<Mutex<LruCache<(String, usize), CachedGraph>>> =
    Lazy::new(|| Mutex::new(LruCache::new(NonZeroUsize::new(1024).unwrap())));

/// Remove cached graphs for `token` or clear the entire cache when `None`.
pub fn invalidate_edges(token: Option<&str>) {
    let mut cache = EDGE_CACHE.lock().unwrap();
    if let Some(tok) = token {
        let keys: Vec<(String, usize)> = cache
            .iter()
            .filter(|(k, _)| k.0 == tok)
            .map(|(k, _)| (k.0.clone(), k.1))
            .collect();
        for k in keys {
            cache.pop(&k);
        }
    } else {
        cache.clear();
    }
}

use crate::TokenInfo;

#[derive(Debug)]
pub struct RouteResult {
    pub path: Vec<String>,
    pub profit: f64,
    pub slippage: f64,
}

fn build_graph(
    dex_map: &HashMap<String, HashMap<String, TokenInfo>>,
    token: &str,
) -> Option<CachedGraph> {
    let mut prices: HashMap<String, f64> = HashMap::new();
    let mut bids: HashMap<String, f64> = HashMap::new();
    let mut asks: HashMap<String, f64> = HashMap::new();
    for (dex, tokens) in dex_map.iter() {
        if let Some(info) = tokens.get(token) {
            let price = if info.asks > 0.0 && info.bids > 0.0 {
                (info.asks + info.bids) / 2.0
            } else if info.asks > 0.0 {
                info.asks
            } else {
                info.bids
            };
            prices.insert(dex.clone(), price);
            bids.insert(dex.clone(), info.bids);
            asks.insert(dex.clone(), info.asks);
        }
    }
    if prices.len() < 2 {
        return None;
    }
    let venues: Vec<String> = prices.keys().cloned().collect();
    let max_price = prices.values().fold(f64::MIN, |a, v| a.max(*v));
    let min_price = prices.values().fold(f64::MAX, |a, v| a.min(*v));
    let max_diff = (max_price - min_price).abs();
    let mut adjacency: AdjMap = HashMap::new();
    for a in venues.iter() {
        let mut neigh = HashMap::new();
        for b in venues.iter() {
            if a == b {
                continue;
            }
            let buy_price = prices.get(a).copied().unwrap_or(0.0);
            let sell_price = prices.get(b).copied().unwrap_or(0.0);
            let ask_liq = asks.get(a).copied().unwrap_or(0.0).max(1.0);
            let bid_liq = bids.get(b).copied().unwrap_or(0.0).max(1.0);
            let slip_rate = 1.0 / ask_liq + 1.0 / bid_liq;
            let profit_rate = (sell_price - buy_price) - buy_price * slip_rate;
            neigh.insert(b.clone(), (profit_rate, slip_rate));
        }
        adjacency.insert(a.clone(), neigh);
    }
    Some(CachedGraph {
        adjacency,
        max_diff,
        venues,
    })
}

fn search_graph(graph: &CachedGraph, amount: f64, max_hops: usize) -> Option<RouteResult> {
    let mut best_path: Option<Vec<String>> = None;
    let mut best_profit = f64::NEG_INFINITY;
    let mut best_slip = 0.0;
    let mut paths: Vec<(String, Vec<String>, f64, f64)> = graph
        .venues
        .iter()
        .map(|v| (v.clone(), vec![v.clone()], 0.0, 0.0))
        .collect();
    for _ in 1..max_hops {
        let mut new_paths = Vec::new();
        for (last, path, profit, slip) in paths.iter() {
            let remaining = max_hops.saturating_sub(path.len());
            if *profit + (remaining as f64) * graph.max_diff * amount < best_profit {
                continue;
            }
            if let Some(neigh) = graph.adjacency.get(last) {
                for (nxt, (profit_rate, slip_rate)) in neigh.iter() {
                    if path.contains(nxt) {
                        continue;
                    }
                    let p = profit_rate * amount;
                    let s = slip_rate * amount;
                    let new_profit = profit + p;
                    let new_slip = slip + s;
                    let mut new_path = path.clone();
                    new_path.push(nxt.clone());
                    if new_profit > best_profit {
                        best_profit = new_profit;
                        best_path = Some(new_path.clone());
                        best_slip = new_slip;
                    }
                    if new_profit
                        + (max_hops.saturating_sub(new_path.len()) as f64) * graph.max_diff * amount
                        >= best_profit
                    {
                        new_paths.push((nxt.clone(), new_path, new_profit, new_slip));
                    }
                }
            }
        }
        paths = new_paths;
        if paths.is_empty() {
            break;
        }
    }
    best_path.map(|p| RouteResult {
        path: p,
        profit: best_profit,
        slippage: best_slip,
    })
}
pub fn best_route(
    dex_map: &HashMap<String, HashMap<String, TokenInfo>>,
    token: &str,
    amount: f64,
    max_hops: usize,
) -> Option<RouteResult> {
    if max_hops < 2 {
        return None;
    }
    let key = (token.to_string(), max_hops);
    let cached = {
        let mut cache = EDGE_CACHE.lock().unwrap();
        cache.get(&key).cloned()
    };
    let graph = if let Some(g) = cached {
        g
    } else {
        let g = build_graph(dex_map, token)?;
        EDGE_CACHE.lock().unwrap().put(key, g.clone());
        g
    };
    search_graph(&graph, amount, max_hops)
}
