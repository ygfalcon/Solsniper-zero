use std::collections::HashMap;

use crate::TokenInfo;

#[derive(Debug)]
pub struct RouteResult {
    pub path: Vec<String>,
    pub profit: f64,
    pub slippage: f64,
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
    let mut best_path: Option<Vec<String>> = None;
    let mut best_profit = f64::NEG_INFINITY;
    let mut best_slip = 0.0;
    let mut paths: Vec<(String, Vec<String>, f64, f64)> = venues
        .iter()
        .map(|v| (v.clone(), vec![v.clone()], 0.0, 0.0))
        .collect();
    for _ in 1..max_hops {
        let mut new_paths = Vec::new();
        for (last, path, profit, slip) in paths.iter() {
            for nxt in venues.iter() {
                if path.contains(nxt) {
                    continue;
                }
                let buy_price = prices.get(last).copied().unwrap_or(0.0);
                let sell_price = prices.get(nxt).copied().unwrap_or(0.0);
                let p = (sell_price - buy_price) * amount;
                let ask_liq = asks.get(last).copied().unwrap_or(0.0).max(1.0);
                let bid_liq = bids.get(nxt).copied().unwrap_or(0.0).max(1.0);
                let slip_val = amount / ask_liq + amount / bid_liq;
                let new_profit = profit + p - slip_val * buy_price;
                let new_slip = slip + slip_val;
                let mut new_path = path.clone();
                new_path.push(nxt.clone());
                if new_profit > best_profit {
                    best_profit = new_profit;
                    best_path = Some(new_path.clone());
                    best_slip = new_slip;
                }
                new_paths.push((nxt.clone(), new_path, new_profit, new_slip));
            }
        }
        paths = new_paths;
        if paths.is_empty() {
            break;
        }
    }
    best_path.map(|p| RouteResult { path: p, profit: best_profit, slippage: best_slip })
}
