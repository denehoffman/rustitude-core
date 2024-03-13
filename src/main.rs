use std::time::Instant;

use nalgebra::{SMatrix, SVector};
use num_complex::Complex64;
use rayon::prelude::*;
use rustitude::{
    four_momentum::FourMomentum,
    gluex::{
        calculate_k_matrix_f0, open_gluex_polarized, precalculate_k_matrix_f0, zlm, Reflectivity,
        Wave,
    },
};
use std::hint::black_box;

fn main() {
    let momentum1 = FourMomentum::new(1.0, 2.0, 3.0, 4.0);
    let momentum2 = FourMomentum::new(5.0, 6.0, 7.0, 8.0);

    let start = Instant::now();
    for _ in 0..1000 {
        let result = momentum1 + momentum2;
        black_box(result);
    }
    println!("Elapsed time: {:.2?}", start.elapsed());
    // let start = Instant::now();
    // let ds = open_gluex_polarized("data_pol.parquet");
    // let duration = start.elapsed();
    // println!("Elapsed: {:?}", duration);
    //
    // let start = Instant::now();
    // let kmat_f0_precalcs: Vec<(SMatrix<Complex64, 5, 5>, SVector<Complex64, 5>)> = ds
    //     .events
    //     .par_iter()
    //     .map(|event| precalculate_k_matrix_f0(event, 2))
    //     .collect();
    // let duration = start.elapsed();
    // println!("Elapsed: {:?}", duration);
    //
    // let betas = SVector::<Complex64, 5>::new(
    //     Complex64::new(1.0, 0.0),
    //     Complex64::new(1.0, 0.0),
    //     Complex64::new(1.0, 0.0),
    //     Complex64::new(1.0, 0.0),
    //     Complex64::new(1.0, 0.0),
    // );
    //
    // for _ in 0..3 {
    //     let start = Instant::now();
    //     let kmat_f0: Vec<Complex64> = ds
    //         .events
    //         .par_iter()
    //         .zip(&kmat_f0_precalcs)
    //         .map(|(event, precalcs)| calculate_k_matrix_f0(event, &betas, &precalcs.0, &precalcs.1))
    //         .collect();
    //     let duration = start.elapsed();
    //     println!("Elapsed: {:?}", duration);
    //     println!("Does it work?: {}", kmat_f0[0]);
    // }
}
