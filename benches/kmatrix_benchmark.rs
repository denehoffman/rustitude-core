use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{SMatrix, SVector};
use num_complex::Complex64;
use rayon::prelude::*;
use rustitude::gluex::{
    calculate_k_matrix, open_gluex_pol_in_beam_parquet, precalculate_k_matrix_f0,
};

fn kmatrix_benchmark(c: &mut Criterion) {
    let ds = open_gluex_pol_in_beam_parquet("data_pol.parquet");
    let kmat_f0_precalcs: Vec<(SVector<Complex64, 5>, SMatrix<Complex64, 5, 5>)> = ds
        .events
        .par_iter()
        .map(|event| precalculate_k_matrix_f0(event, 2))
        .collect();

    let betas = SVector::<Complex64, 5>::new(
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    );

    c.bench_function("K-Matrix F0", |b| {
        b.iter(|| {
            let kmat_f0: Vec<Complex64> = ds
                .events
                .par_iter()
                .zip(&kmat_f0_precalcs)
                .map(|(event, precalcs)| calculate_k_matrix(event, &betas, precalcs))
                .collect();
            black_box(kmat_f0);
        })
    });
}

fn kmatrix_benchmark_mc(c: &mut Criterion) {
    let ds = open_gluex_pol_in_beam_parquet("accmc_pol.parquet");
    let kmat_f0_precalcs: Vec<(SVector<Complex64, 5>, SMatrix<Complex64, 5, 5>)> = ds
        .events
        .par_iter()
        .map(|event| precalculate_k_matrix_f0(event, 2))
        .collect();

    let betas = SVector::<Complex64, 5>::new(
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    );

    c.bench_function("K-Matrix F0 (MC)", |b| {
        b.iter(|| {
            let kmat_f0: Vec<Complex64> = ds
                .events
                .par_iter()
                .zip(&kmat_f0_precalcs)
                .map(|(event, precalcs)| calculate_k_matrix(event, &betas, precalcs))
                .collect();
            black_box(kmat_f0);
        })
    });
}

criterion_group!(benches, kmatrix_benchmark, kmatrix_benchmark_mc);
criterion_main!(benches);
