use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(feature = "heuristic")]
use route_ffi::best_route_heuristic;
use route_ffi::best_route_internal;
use std::collections::HashMap;

fn build_data() -> (
    HashMap<String, f64>,
    HashMap<String, f64>,
    HashMap<String, f64>,
    HashMap<String, f64>,
) {
    let mut prices = HashMap::new();
    let mut fees = HashMap::new();
    let mut gas = HashMap::new();
    let mut lat = HashMap::new();
    for i in 0..10 {
        let name = format!("dex{}", i);
        prices.insert(name.clone(), 1.0 + i as f64 * 0.01);
        fees.insert(name.clone(), 0.001);
        gas.insert(name.clone(), 0.0);
        lat.insert(name.clone(), 0.0);
    }
    (prices, fees, gas, lat)
}

fn bench_internal(c: &mut Criterion) {
    let (p, f, g, l) = build_data();
    c.bench_function("internal", |b| {
        b.iter(|| {
            let _ = best_route_internal(&p, 1.0, &f, &g, &l, 4);
        });
    });
}

#[cfg(feature = "heuristic")]
fn bench_heuristic(c: &mut Criterion) {
    let (p, f, g, l) = build_data();
    c.bench_function("heuristic", |b| {
        b.iter(|| {
            let _ = best_route_heuristic(&p, 1.0, &f, &g, &l, 4);
        });
    });
}

#[cfg(feature = "heuristic")]
criterion_group!(benches, bench_internal, bench_heuristic);
#[cfg(not(feature = "heuristic"))]
criterion_group!(benches, bench_internal);
criterion_main!(benches);
