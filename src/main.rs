use std::time::Instant;

use nalgebra::{SMatrix, SVector};
use num_complex::Complex64;
use rayon::prelude::*;
use rustitude::gluex::{
    calculate_k_matrix_f0, open_gluex_polarized, precalculate_k_matrix_f0, zlm, Reflectivity, Wave,
};

fn main() {
    let start = Instant::now();
    let ds = open_gluex_polarized("data_pol.parquet");
    let duration = start.elapsed();
    println!("Elapsed: {:?}", duration);

    let start = Instant::now();
    let zlm_s0p: Vec<Complex64> = ds
        .events
        .par_iter()
        .map(|event| zlm(event, Wave::S0, Reflectivity::Positive))
        .collect();
    let duration = start.elapsed();
    println!("Elapsed: {:?}", duration);

    let start = Instant::now();
    let zlm_s0n: Vec<Complex64> = ds
        .events
        .par_iter()
        .map(|event| zlm(event, Wave::S0, Reflectivity::Negative))
        .collect();
    let duration = start.elapsed();
    println!("Elapsed: {:?}", duration);

    let start = Instant::now();
    let zlm_d2p: Vec<Complex64> = ds
        .events
        .par_iter()
        .map(|event| zlm(event, Wave::D2, Reflectivity::Positive))
        .collect();
    let duration = start.elapsed();
    println!("Elapsed: {:?}", duration);

    let start = Instant::now();
    let kmat_f0_precalcs: Vec<(SMatrix<Complex64, 5, 5>, SVector<Complex64, 5>)> = ds
        .events
        .par_iter()
        .map(|event| precalculate_k_matrix_f0(event, 2))
        .collect();
    let duration = start.elapsed();
    println!("Elapsed: {:?}", duration);

    let betas = SVector::<Complex64, 5>::new(
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    );
    let start = Instant::now();
    let kmat_f0: Vec<Complex64> = ds
        .events
        .par_iter()
        .zip(kmat_f0_precalcs)
        .map(|(event, precalcs)| calculate_k_matrix_f0(event, &betas, &precalcs.0, &precalcs.1))
        .collect();
    let duration = start.elapsed();
    println!("Elapsed: {:?}", duration);
    println!("Does it work?: {}", kmat_f0[0]);

    let start = Instant::now();
    let a = Complex64::new(1.0, 0.0);
    let b = Complex64::new(1.0, 0.0);
    let c = Complex64::new(1.0, 0.0);
    let res: Vec<f64> = (zlm_s0p, zlm_s0n, zlm_d2p)
        .par_iter()
        .map(|(s0p, s0n, d2p)| {
            (a * s0p.re + b * d2p.re).norm_sqr()
                + (a * s0p.im + b * d2p.im).norm_sqr()
                + (c * s0n.re).norm_sqr()
                + (c * s0n.im).norm_sqr()
        })
        .collect();
    let duration = start.elapsed();
    println!("Elapsed: {:?}", duration);
    println!("{}", res[10]);
}
