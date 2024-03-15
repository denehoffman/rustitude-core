use std::{hint::black_box, time::Instant};

use nalgebra::{SMatrix, SVector};
use num_complex::Complex64;
use rayon::prelude::*;
use rustitude::gluex::{
    calculate_k_matrix, open_gluex_pol_in_beam_parquet, precalculate_k_matrix_f0,
};

fn main() {
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

    for _ in 0..5 {
        let start = Instant::now();
        let kmat_f0: Vec<Complex64> = ds
            .events
            .par_iter()
            .zip(&kmat_f0_precalcs)
            .map(|(event, precalcs)| calculate_k_matrix(event, &betas, precalcs))
            .collect();
        black_box(kmat_f0);
        println!("Elapsed: {:?}", start.elapsed());
    }
}
