use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_complex::Complex64;
use rand::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use rustitude::gluex::{open_gluex_pol_in_beam_parquet, KMatrixF0};
use rustitude::prelude::*;
use uuid::Uuid;

fn kmatrix_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let ds = open_gluex_pol_in_beam_parquet("data_pol.parquet");
    let mut f0_kmatrix = KMatrixF0::default();
    let mut f0_amplitude = Amplitude::new(&ds, &mut f0_kmatrix);
    f0_amplitude.precompute();
    let mut f0_kmatrix_params: HashMap<String, f64> = HashMap::default();
    f0_kmatrix_params.insert("f0_500 re".to_string(), rng.gen());
    f0_kmatrix_params.insert("f0_500 im".to_string(), rng.gen());
    f0_kmatrix_params.insert("f0_980 re".to_string(), rng.gen());
    f0_kmatrix_params.insert("f0_980 im".to_string(), rng.gen());
    f0_kmatrix_params.insert("f0_1370 re".to_string(), rng.gen());
    f0_kmatrix_params.insert("f0_1370 im".to_string(), rng.gen());
    f0_kmatrix_params.insert("f0_1500 re".to_string(), rng.gen());
    f0_kmatrix_params.insert("f0_1500 im".to_string(), rng.gen());
    f0_kmatrix_params.insert("f0_1710 re".to_string(), rng.gen());
    f0_kmatrix_params.insert("f0_1710 im".to_string(), rng.gen());
    let mut params: HashMap<Uuid, HashMap<String, f64>> = HashMap::default();
    params.insert(f0_amplitude.uuid, f0_kmatrix_params);

    c.bench_function("K-Matrix F0", |b| {
        b.iter(|| {
            let mut f0_kmatrix_params: HashMap<String, f64> = HashMap::default();
            f0_kmatrix_params.insert("f0_500 re".to_string(), rng.gen());
            f0_kmatrix_params.insert("f0_500 im".to_string(), rng.gen());
            f0_kmatrix_params.insert("f0_980 re".to_string(), rng.gen());
            f0_kmatrix_params.insert("f0_980 im".to_string(), rng.gen());
            f0_kmatrix_params.insert("f0_1370 re".to_string(), rng.gen());
            f0_kmatrix_params.insert("f0_1370 im".to_string(), rng.gen());
            f0_kmatrix_params.insert("f0_1500 re".to_string(), rng.gen());
            f0_kmatrix_params.insert("f0_1500 im".to_string(), rng.gen());
            f0_kmatrix_params.insert("f0_1710 re".to_string(), rng.gen());
            f0_kmatrix_params.insert("f0_1710 im".to_string(), rng.gen());
            params.insert(f0_amplitude.uuid, f0_kmatrix_params);

            let kmat_f0: Vec<Complex64> = f0_amplitude.compute(&params);
            black_box(kmat_f0);
        })
    });
}

fn kmatrix_benchmark_mc(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let ds = open_gluex_pol_in_beam_parquet("accmc_pol.parquet");
    let mut f0_kmatrix = KMatrixF0::default();
    let mut f0_amplitude = Amplitude::new(&ds, &mut f0_kmatrix);
    f0_amplitude.precompute();
    let mut f0_kmatrix_params: HashMap<String, f64> = HashMap::default();
    f0_kmatrix_params.insert("f0_500 re".to_string(), rng.gen());
    f0_kmatrix_params.insert("f0_500 im".to_string(), rng.gen());
    f0_kmatrix_params.insert("f0_980 re".to_string(), rng.gen());
    f0_kmatrix_params.insert("f0_980 im".to_string(), rng.gen());
    f0_kmatrix_params.insert("f0_1370 re".to_string(), rng.gen());
    f0_kmatrix_params.insert("f0_1370 im".to_string(), rng.gen());
    f0_kmatrix_params.insert("f0_1500 re".to_string(), rng.gen());
    f0_kmatrix_params.insert("f0_1500 im".to_string(), rng.gen());
    f0_kmatrix_params.insert("f0_1710 re".to_string(), rng.gen());
    f0_kmatrix_params.insert("f0_1710 im".to_string(), rng.gen());
    let mut params: HashMap<Uuid, HashMap<String, f64>> = HashMap::default();
    params.insert(f0_amplitude.uuid, f0_kmatrix_params);

    c.bench_function("K-Matrix F0 (MC)", |b| {
        b.iter(|| {
            let mut f0_kmatrix_params: HashMap<String, f64> = HashMap::default();
            f0_kmatrix_params.insert("f0_500 re".to_string(), rng.gen());
            f0_kmatrix_params.insert("f0_500 im".to_string(), rng.gen());
            f0_kmatrix_params.insert("f0_980 re".to_string(), rng.gen());
            f0_kmatrix_params.insert("f0_980 im".to_string(), rng.gen());
            f0_kmatrix_params.insert("f0_1370 re".to_string(), rng.gen());
            f0_kmatrix_params.insert("f0_1370 im".to_string(), rng.gen());
            f0_kmatrix_params.insert("f0_1500 re".to_string(), rng.gen());
            f0_kmatrix_params.insert("f0_1500 im".to_string(), rng.gen());
            f0_kmatrix_params.insert("f0_1710 re".to_string(), rng.gen());
            f0_kmatrix_params.insert("f0_1710 im".to_string(), rng.gen());
            params.insert(f0_amplitude.uuid, f0_kmatrix_params);

            let kmat_f0: Vec<Complex64> = f0_amplitude.compute(&params);
            black_box(kmat_f0);
        })
    });
}

criterion_group! {name = benches; config = Criterion::default().sample_size(100); targets = kmatrix_benchmark, kmatrix_benchmark_mc}
criterion_main!(benches);
