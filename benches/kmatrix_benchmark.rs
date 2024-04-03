use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use rustitude::{amplitude::Manager, gluex::KMatrixF0, prelude::*};

fn kmatrix_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let ds = Dataset::from_parquet("data_pol.parquet", false);
    let mut a_kmatrix_f0 = Amplitude::new("KMatrix[F0]", KMatrixF0::default());
    let mut m = Manager::new(&ds);
    a_kmatrix_f0.precompute(&ds);
    let arca_kmatrix_f0 = Arc::new(a_kmatrix_f0);
    m.register("sum", "F0", &arca_kmatrix_f0);

    c.bench_function("K-Matrix F0", |b| {
        b.iter(|| {
            let res: Vec<f64> = m.compute(vec![rng.gen(); 10]);
            black_box(res);
        })
    });
}

fn kmatrix_benchmark_mc(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let ds = Dataset::from_parquet("accmc_pol.parquet", false);
    let mut a_kmatrix_f0 = Amplitude::new("KMatrix[F0]", KMatrixF0::default());
    let mut m = Manager::new(&ds);
    a_kmatrix_f0.precompute(&ds);
    let arca_kmatrix_f0 = Arc::new(a_kmatrix_f0);
    m.register("sum", "F0", &arca_kmatrix_f0);

    c.bench_function("K-Matrix F0 (MC)", |b| {
        b.iter(|| {
            let res: Vec<f64> = m.compute(vec![rng.gen(); 10]);
            black_box(res);
        })
    });
}

criterion_group! {name = benches; config = Criterion::default().sample_size(100); targets = kmatrix_benchmark, kmatrix_benchmark_mc}
criterion_main!(benches);
