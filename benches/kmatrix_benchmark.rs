use std::sync::{Arc, RwLock};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use rustitude::{
    amplitude::Manager,
    gluex::{ImZlm, KMatrixF0, ReZlm, Reflectivity, Wave},
    prelude::*,
};

fn zlm_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let ds = Dataset::from_parquet("data_pol.parquet", true);
    let rezlm_s0p = Arc::new(RwLock::new(Amplitude::new(
        "Name: Re[Zlm(S0+)]",
        ReZlm::new(Wave::S0, Reflectivity::Positive),
    )));
    let imzlm_s0p = Arc::new(RwLock::new(Amplitude::new(
        "Name: Im[Zlm(S0+)]",
        ImZlm::new(Wave::S0, Reflectivity::Positive),
    )));
    let rezlm_s0n = Arc::new(RwLock::new(Amplitude::new(
        "Name: Re[Zlm(S0-)]",
        ReZlm::new(Wave::S0, Reflectivity::Negative),
    )));
    let imzlm_s0n = Arc::new(RwLock::new(Amplitude::new(
        "Name: Im[Zlm(S0-)]",
        ImZlm::new(Wave::S0, Reflectivity::Negative),
    )));
    let rezlm_d2p = Arc::new(RwLock::new(Amplitude::new(
        "Name: Re[Zlm(D2+)]",
        ReZlm::new(Wave::D2, Reflectivity::Positive),
    )));
    let imzlm_d2p = Arc::new(RwLock::new(Amplitude::new(
        "Name: Im[Zlm(D2+)]",
        ImZlm::new(Wave::D2, Reflectivity::Positive),
    )));
    let mut m = Manager::new(&ds);
    m.register("Sum: PosRe", "Group: S0+", &rezlm_s0p);
    m.register("Sum: PosIm", "Group: S0+", &imzlm_s0p);
    m.register("Sum: NegRe", "Group: S0-", &rezlm_s0n);
    m.register("Sum: NegIm", "Group: S0-", &imzlm_s0n);
    m.register("Sum: PosRe", "Group: D2+", &rezlm_d2p);
    m.register("Sum: PosIm", "Group: D2+", &imzlm_d2p);
    m.precompute();
    m.constrain(
        ("Sum: PosRe", "Group: S0+", "Name: Re[Zlm(S0+)]", "Re"),
        ("Sum: PosIm", "Group: S0+", "Name: Im[Zlm(S0+)]", "Re"),
    );
    m.constrain(
        ("Sum: PosRe", "Group: S0+", "Name: Re[Zlm(S0+)]", "Im"),
        ("Sum: PosIm", "Group: S0+", "Name: Im[Zlm(S0+)]", "Im"),
    );
    m.constrain(
        ("Sum: NegRe", "Group: S0-", "Name: Re[Zlm(S0-)]", "Re"),
        ("Sum: NegIm", "Group: S0-", "Name: Im[Zlm(S0-)]", "Re"),
    );
    m.constrain(
        ("Sum: NegRe", "Group: S0-", "Name: Re[Zlm(S0-)]", "Im"),
        ("Sum: NegIm", "Group: S0-", "Name: Im[Zlm(S0-)]", "Im"),
    );
    m.constrain(
        ("Sum: PosRe", "Group: D2+", "Name: Re[Zlm(D2+)]", "Re"),
        ("Sum: PosIm", "Group: D2+", "Name: Im[Zlm(D2+)]", "Re"),
    );
    m.constrain(
        ("Sum: PosRe", "Group: D2+", "Name: Re[Zlm(D2+)]", "Im"),
        ("Sum: PosIm", "Group: D2+", "Name: Im[Zlm(D2+)]", "Im"),
    );
    c.bench_function("Zlm Benchmark", |b| {
        b.iter(|| {
            let res: Vec<f64> = m.compute(&[rng.gen(); 6]);
            black_box(res);
        })
    });
}

fn kmatrix_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let ds = Dataset::from_parquet("data_pol.parquet", false);
    let kmatrix_f0 = Arc::new(RwLock::new(Amplitude::new(
        "KMatrix[F0]",
        KMatrixF0::default(),
    )));
    let mut m = Manager::new(&ds);
    m.register("sum", "F0", &kmatrix_f0);
    m.precompute();
    c.bench_function("K-Matrix F0", |b| {
        b.iter(|| {
            let res: Vec<f64> = m.compute(&[rng.gen(); 10]);
            black_box(res);
        })
    });
}

fn kmatrix_benchmark_mc(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let ds = Dataset::from_parquet("accmc_pol.parquet", false);
    let kmatrix_f0 = Arc::new(RwLock::new(Amplitude::new(
        "KMatrix[F0]",
        KMatrixF0::default(),
    )));
    let mut m = Manager::new(&ds);
    m.register("sum", "F0", &kmatrix_f0);
    m.precompute();
    c.bench_function("K-Matrix F0 (MC)", |b| {
        b.iter(|| {
            let res: Vec<f64> = m.compute(&[rng.gen(); 10]);
            black_box(res);
        })
    });
}

criterion_group! {name = benches; config = Criterion::default().sample_size(100); targets = zlm_benchmark, kmatrix_benchmark, kmatrix_benchmark_mc}
criterion_main!(benches);
