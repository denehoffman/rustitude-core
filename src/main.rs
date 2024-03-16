use std::{hint::black_box, time::Instant};

use rand::prelude::*;
use rustitude::gluex::{open_gluex_pol_in_beam_parquet, KMatrixF0, Reflectivity, Wave, Zlm};
use rustitude::prelude::*;

fn main() {
    let ds = open_gluex_pol_in_beam_parquet("accmc_pol.parquet");
    // let mut f0_kmatrix = KMatrixF0::default();
    let mut zlm_d2p = Zlm::new(Wave::D2, Reflectivity::Positive);
    let mut rng = rand::thread_rng();
    for _ in 0..5 {
        let mut params = ParameterSet::default();
        // params.insert("f0_500 re", rng.gen());
        // params.insert("f0_500 im", rng.gen());
        // params.insert("f0_980 re", rng.gen());
        // params.insert("f0_980 im", rng.gen());
        // params.insert("f0_1370 re", rng.gen());
        // params.insert("f0_1370 im", rng.gen());
        // params.insert("f0_1500 re", rng.gen());
        // params.insert("f0_1500 im", rng.gen());
        // params.insert("f0_1710 re", rng.gen());
        // params.insert("f0_1710 im", rng.gen());
        let start = Instant::now();
        let zlm_res = zlm_d2p.calculate(&params, &ds);
        black_box(zlm_res);
        // let kmat_f0 = f0_kmatrix.calculate(&params, &ds);
        // black_box(kmat_f0);
        println!("Elapsed: {:?}", start.elapsed());
    }
}
