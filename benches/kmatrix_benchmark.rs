use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{SMatrix, SVector};
use rand::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use rustitude::gluex::{open_gluex_pol_in_beam_parquet, GlueXEvent, KMatrixF0};
use rustitude::prelude::*;

fn kmatrix_benchmark(c: &mut Criterion) {
    let ds = open_gluex_pol_in_beam_parquet("data_pol.parquet");
    let ds_converted: Dataset<GlueXEvent> = ds.convert();
    let mut f0_kmatrix = KMatrixF0::default();
    let mut rng = rand::thread_rng();
    let mut params = ParameterSet::default();
    params.insert("f0_500 re", rng.gen());
    params.insert("f0_500 im", rng.gen());
    params.insert("f0_980 re", rng.gen());
    params.insert("f0_980 im", rng.gen());
    params.insert("f0_1370 re", rng.gen());
    params.insert("f0_1370 im", rng.gen());
    params.insert("f0_1500 re", rng.gen());
    params.insert("f0_1500 im", rng.gen());
    params.insert("f0_1710 re", rng.gen());
    params.insert("f0_1710 im", rng.gen());
    let _kmat_f0 = f0_kmatrix.calculate(&params, &ds_converted);
    c.bench_function("K-Matrix F0", |b| {
        b.iter(|| {
            let mut params = ParameterSet::default();
            params.insert("f0_500 re", rng.gen());
            params.insert("f0_500 im", rng.gen());
            params.insert("f0_980 re", rng.gen());
            params.insert("f0_980 im", rng.gen());
            params.insert("f0_1370 re", rng.gen());
            params.insert("f0_1370 im", rng.gen());
            params.insert("f0_1500 re", rng.gen());
            params.insert("f0_1500 im", rng.gen());
            params.insert("f0_1710 re", rng.gen());
            params.insert("f0_1710 im", rng.gen());
            let kmat_f0 = f0_kmatrix.calculate(&params, &ds_converted);
            black_box(kmat_f0);
        })
    });
}

fn kmatrix_benchmark_mc(c: &mut Criterion) {
    let ds = open_gluex_pol_in_beam_parquet("accmc_pol.parquet");
    let ds_converted: Dataset<GlueXEvent> = ds.convert();
    let mut f0_kmatrix = KMatrixF0::default();
    let mut rng = rand::thread_rng();
    let mut params = ParameterSet::default();
    params.insert("f0_500 re", rng.gen());
    params.insert("f0_500 im", rng.gen());
    params.insert("f0_980 re", rng.gen());
    params.insert("f0_980 im", rng.gen());
    params.insert("f0_1370 re", rng.gen());
    params.insert("f0_1370 im", rng.gen());
    params.insert("f0_1500 re", rng.gen());
    params.insert("f0_1500 im", rng.gen());
    params.insert("f0_1710 re", rng.gen());
    params.insert("f0_1710 im", rng.gen());
    let _kmat_f0 = f0_kmatrix.calculate(&params, &ds_converted);
    c.bench_function("K-Matrix F0 (MC)", |b| {
        b.iter(|| {
            let mut params = ParameterSet::default();
            params.insert("f0_500 re", rng.gen());
            params.insert("f0_500 im", rng.gen());
            params.insert("f0_980 re", rng.gen());
            params.insert("f0_980 im", rng.gen());
            params.insert("f0_1370 re", rng.gen());
            params.insert("f0_1370 im", rng.gen());
            params.insert("f0_1500 re", rng.gen());
            params.insert("f0_1500 im", rng.gen());
            params.insert("f0_1710 re", rng.gen());
            params.insert("f0_1710 im", rng.gen());
            let kmat_f0 = f0_kmatrix.calculate(&params, &ds_converted);
            black_box(kmat_f0);
        })
    });
}

criterion_group! {name = benches; config = Criterion::default().sample_size(1000); targets = kmatrix_benchmark, kmatrix_benchmark_mc}
criterion_main!(benches);
