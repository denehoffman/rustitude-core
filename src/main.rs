use std::hint::black_box;

use num_complex::Complex64;
use rand::prelude::*;

use rustc_hash::FxHashMap as HashMap;
use rustitude::amplitude::{Amplitude, Node};
use rustitude::dataset::Event;
use rustitude::gluex::{open_gluex_pol_in_beam_parquet, KMatrixF0, Wave, Ylm};

use uuid::Uuid;

#[derive(Default)]
struct Wrapper<E: Event> {
    nodes: Vec<Box<dyn Node<E>>>,
}

fn main() {
    let ds = open_gluex_pol_in_beam_parquet("data_pol.parquet");
    let mut f0_kmatrix = KMatrixF0::default();
    let mut ylm00 = Ylm::new(Wave::S0);
    ylm00.precalculate(&ds);
    f0_kmatrix.precalculate(&ds);
    let a = ylm00.calculate(0, ds.iter().next().unwrap(), &Vec::new());
    let b = f0_kmatrix.calculate(0, ds.iter().next().unwrap(), &vec![1.0; 10]);
    let mut w = Wrapper::default();
    w.nodes.push(Box::new(f0_kmatrix));
    // let mut f0_amplitude = Amplitude::new(&ds, &mut f0_kmatrix);
    // f0_amplitude.precompute();
    // let mut rng = rand::thread_rng();
    // let mut f0_kmatrix_params: HashMap<String, f64> = HashMap::default();
    // f0_kmatrix_params.insert("f0_500 re".to_string(), rng.gen());
    // f0_kmatrix_params.insert("f0_500 im".to_string(), rng.gen());
    // f0_kmatrix_params.insert("f0_980 re".to_string(), rng.gen());
    // f0_kmatrix_params.insert("f0_980 im".to_string(), rng.gen());
    // f0_kmatrix_params.insert("f0_1370 re".to_string(), rng.gen());
    // f0_kmatrix_params.insert("f0_1370 im".to_string(), rng.gen());
    // f0_kmatrix_params.insert("f0_1500 re".to_string(), rng.gen());
    // f0_kmatrix_params.insert("f0_1500 im".to_string(), rng.gen());
    // f0_kmatrix_params.insert("f0_1710 re".to_string(), rng.gen());
    // f0_kmatrix_params.insert("f0_1710 im".to_string(), rng.gen());
    // let mut params: HashMap<Uuid, HashMap<String, f64>> = HashMap::default();
    // params.insert(f0_amplitude.uuid, f0_kmatrix_params);
    // for _ in 0..200 {
    //     let kmat_f0: Vec<Complex64> = f0_amplitude.compute(&params);
    //     black_box(kmat_f0);
    // }
}
