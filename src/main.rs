#![allow(unused_imports)]
#![allow(unreachable_code)]
use argmin::solver::neldermead::NelderMead;
use rand::Rng;
use rustc_hash::FxHashMap as HashMap;
use std::time::Instant;

use argmin::core::observers::{ObserverMode, SlogLogger};
use argmin::core::{CostFunction, Executor, State};
use argmin::solver::particleswarm::ParticleSwarm;
use ndarray::array;
use num_complex::Complex64;
use num_traits::Pow;
use rayon::prelude::*;
// use rustitude::gluex::KMatrixConstants;
use rustitude::gluex::{
    open_gluex, AdlerZero, FrozenKMatrix, HelicityVec, KMatrixConstants, Reflectivity,
    ResonanceMass, Wave, Ylm, Zlm,
};
use rustitude::prelude::*;

fn simplex(input: Vec<f64>, step: f64) -> Vec<Vec<f64>> {
    let mut result = Vec::new();
    result.push(input.clone());
    for i in 0..input.len() {
        let mut new_vertex = input.clone();
        new_vertex[i] += step;
        result.push(new_vertex);
    }
    result
}
fn generate_bin_edges(min: f64, max: f64, num_bins: usize) -> Vec<(f64, f64)> {
    (0..num_bins)
        .map(|i| {
            (
                min + i as f64 * (max - min) / num_bins as f64,
                min + (i + 1) as f64 * (max - min) / num_bins as f64,
            )
        })
        .collect()
}
fn main() {
    let zlm_s0_plus = Zlm::new(Wave::S0, Reflectivity::Positive);
    let zlm_d2_plus = Zlm::new(Wave::D2, Reflectivity::Positive);
    let par_s0_plus = ComplexParameterNode::new("S0+ re", "S0+ im");
    let par_d2_plus = ComplexParameterNode::new("D2+ re", "D2+ im");
    let pos_re = zlm_s0_plus
        .real()
        .mul(&par_s0_plus)
        .add(&zlm_d2_plus.real().mul(&par_d2_plus));

    let pos_im = zlm_s0_plus
        .imag()
        .mul(&par_s0_plus)
        .add(&zlm_d2_plus.imag().mul(&par_d2_plus));

    let amp = pos_re.norm_sqr().add(&pos_im.norm_sqr());

    let edges = generate_bin_edges(1.0, 2.0, 40);
    let now = Instant::now();
    for bin in &edges {
        let dataset = open_gluex("data_pol.parquet", true, *bin).unwrap();
        let montecarlo = open_gluex("accmc_pol.parquet", true, *bin).unwrap();
        let eml = EML::new(
            dataset,
            montecarlo,
            Box::new(amp.clone()),
            Box::new(amp.clone()),
            vec![
                par!("S0+ re"),
                par!("S0+ im", 0.0),
                par!("D2+ re"),
                par!("D2+ im"),
            ],
        );
        let solver = NelderMead::new(simplex(vec![100.0, 100.0, 100.0], 10.0));
        let res = Executor::new(eml, solver)
            .configure(|state| state.max_iters(400))
            .run()
            .unwrap();
        let best_vec = res.state().get_best_param().unwrap();
        println!(
            "Bin [{}, {}]: (S0+ re, D2+ re, D2+ im) = ({}, {}, {})",
            bin.0, bin.1, best_vec[0], best_vec[1], best_vec[2]
        );
    }
    let elapsed = now.elapsed();
    println!("Total time: {:.2?}", elapsed);

    // KMatrix
    // let f0 = KMatrixConstants {
    //     name: "f0".to_string(),
    //     res_names: vec![
    //         "f0_500".to_string(),
    //         "f0_980".to_string(),
    //         "f0_1370".to_string(),
    //         "f0_1500".to_string(),
    //         "f0_1710".to_string(),
    //     ],
    //     g: array![
    //         [0.74987, 0.06401, -0.23417, 0.0157, -0.14242],
    //         [-0.01257, 0.00204, -0.01032, 0.267, 0.2278],
    //         [0.02736, 0.77413, 0.72283, 0.09214, 0.15981],
    //         [-0.15102, 0.50999, 0.11934, 0.02742, 0.16272],
    //         [0.36103, 0.13112, 0.36792, -0.04025, -0.17397]
    //     ],
    //     m: array![0.51461, 0.90630, 1.23089, 1.46104, 1.69611],
    //     c: array![
    //         [0.03728, 0.00000, -0.01398, -0.02203, 0.01397],
    //         [0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
    //         [-0.01398, 0.00000, 0.02349, 0.03101, -0.04003],
    //         [-0.02203, 0.00000, 0.03101, -0.13769, -0.06722],
    //         [0.01397, 0.00000, -0.04003, -0.06722, -0.28401],
    //     ],
    //     m1: array![0.13498, 0.26995, 0.49368, 0.54786, 0.54786],
    //     m2: array![0.13498, 0.26995, 0.49761, 0.54786, 0.95778],
    //     n_resonances: 5,
    //     n_channels: 5,
    //     wave: Wave::S,
    //     adler_zero: Some(AdlerZero {
    //         s_0: 0.009_112_5,
    //         s_norm: 1.0,
    //     }),
    // };
    //
    // let f2 = KMatrixConstants {
    //     name: "f2".to_string(),
    //     res_names: vec![
    //         "f2_1270".to_string(),
    //         "f2_1525".to_string(),
    //         "f2_1810".to_string(),
    //         "f2_1950".to_string(),
    //     ],
    //     g: array![
    //         [0.40033, 0.0182, -0.06709, -0.49924],
    //         [0.15479, 0.173, 0.22941, 0.19295],
    //         [-0.089, 0.32393, -0.43133, 0.27975],
    //         [-0.00113, 0.15256, 0.23721, -0.03987]
    //     ],
    //     m: array![1.15299, 1.48359, 1.72923, 1.96700],
    //     c: array![
    //         [-0.04319, 0.00000, 0.00984, 0.01028],
    //         [0.00000, 0.00000, 0.00000, 0.00000],
    //         [0.00984, 0.00000, -0.07344, 0.05533],
    //         [0.01028, 0.00000, 0.05533, -0.05183],
    //     ],
    //     m1: array![0.13498, 0.26995, 0.49368, 0.54786],
    //     m2: array![0.13498, 0.26995, 0.49761, 0.54786],
    //     n_resonances: 4,
    //     n_channels: 4,
    //     wave: Wave::D,
    //     adler_zero: None,
    // };
    //
    // let a0 = KMatrixConstants {
    //     name: "a0".to_string(),
    //     res_names: vec!["a0_980".to_string(), "a0_1450".to_string()],
    //     g: array![[0.43215, 0.19], [-0.28825, 0.43372]],
    //     m: array![0.95395, 1.26767],
    //     c: array![[0.00000, 0.00000], [0.00000, 0.00000],],
    //     m1: array![0.13498, 0.49368],
    //     m2: array![0.54786, 0.49761],
    //     n_resonances: 2,
    //     n_channels: 2,
    //     wave: Wave::S,
    //     adler_zero: None,
    // };
    //
    // let a2 = KMatrixConstants {
    //     name: "a2".to_string(),
    //     res_names: vec!["a2_1320".to_string(), "a2_1700".to_string()],
    //     g: array![[0.30073, 0.68567], [0.21426, 0.12543], [-0.09162, 0.00184]],
    //     m: array![1.30080, 1.75351],
    //     c: array![
    //         [-0.40184, 0.00033, -0.08707],
    //         [0.00033, -0.21416, -0.06193],
    //         [-0.08707, -0.06193, -0.17435],
    //     ],
    //     m1: array![0.13498, 0.49368, 0.13498],
    //     m2: array![0.54786, 0.49761, 0.95778],
    //     n_resonances: 2,
    //     n_channels: 2,
    //     wave: Wave::D,
    //     adler_zero: None,
    // };
    //
    // let f0_amp = FrozenKMatrix::new("ksks", 2, f0);
    // let f2_amp = FrozenKMatrix::new("ksks", 2, f2);
    // let a0_amp = FrozenKMatrix::new("ksks", 1, a0);
    // let a2_amp = FrozenKMatrix::new("ksks", 1, a2);
    //
    // let kmat_pos_re = zlm_s0_plus
    //     .real()
    //     .mul(&f0_amp.add(&a0_amp))
    //     .add(&zlm_d2_plus.real().mul(&f2_amp.add(&a2_amp)))
    //     .norm_sqr();
    //
    // let kmat_pos_im = zlm_s0_plus
    //     .imag()
    //     .mul(&f0_amp.add(&a0_amp))
    //     .add(&zlm_d2_plus.imag().mul(&f2_amp.add(&a2_amp)))
    //     .norm_sqr();
    //
    // let kmat_amp = kmat_pos_re.add(&kmat_pos_im);
    //
    // let kmat_eml = EML::new(
    //     &mut dataset,
    //     &mut montecarlo,
    //     &kmat_amp,
    //     &kmat_amp,
    //     vec![
    //         "ksks_f0_500_re",
    //         "ksks_f0_500_im",
    //         "ksks_f0_980_re",
    //         "ksks_f0_980_im",
    //         "ksks_f0_1370_re",
    //         "ksks_f0_1370_im",
    //         "ksks_f0_1500_re",
    //         "ksks_f0_1500_im",
    //         "ksks_f0_1710_re",
    //         "ksks_f0_1710_im",
    //         "ksks_f2_1270_re",
    //         "ksks_f2_1270_im",
    //         "ksks_f2_1525_re",
    //         "ksks_f2_1525_im",
    //         "ksks_f2_1810_re",
    //         "ksks_f2_1810_im",
    //         "ksks_f2_1950_re",
    //         "ksks_f2_1950_im",
    //         "ksks_a0_980_re",
    //         "ksks_a0_980_im",
    //         "ksks_a0_1450_re",
    //         "ksks_a0_1450_im",
    //         "ksks_a2_1320_re",
    //         "ksks_a2_1320_im",
    //         "ksks_a2_1700_re",
    //         "ksks_a2_1700_im",
    //     ],
    // );
    // let elapsed = now.elapsed();
    // println!("Resolve EML: {:.2?}", elapsed);
    // let now = Instant::now();
    // let res = kmat_eml.cost(&vec![2.3; 26]).unwrap();
    // let elapsed = now.elapsed();
    // println!("Calculate EML: {:.2?}", elapsed);
    // println!("Result: {}", res);
}
