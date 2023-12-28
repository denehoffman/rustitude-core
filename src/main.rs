use std::time::Instant;

use ndarray::array;
use num_complex::Complex64;
use rustitude::gluex;
use rustitude::gluex::KMatrixConstants;
use rustitude::prelude::*;

#[allow(clippy::too_many_lines)]
fn main() {
    let particles = gluex::ParticleInfo {
        recoil_index: 0,
        daughter_index: 1,
        resonance_indices: Vec::from([1, 2]),
    };
    let zlm_00p = gluex::Zlm {
        l: 0,
        m: 0,
        r: gluex::Reflectivity::Positive,
        particle_info: particles.clone(),
    }
    .to_amplitude();

    let zlm_22p = gluex::Zlm {
        l: 2,
        m: 2,
        r: gluex::Reflectivity::Positive,
        particle_info: particles.clone(),
    }
    .to_amplitude();

    let f0_500 = cpar!("f0_500", 0.0, 0.0);
    let f0_980 = cpar!("f0_980", 100.0, 0.0);
    let f0_1370 = cpar!("f0_1370", 50.0, 50.0);
    let f0_1500 = cpar!("f0_1500", 50.0, 50.0);
    let f0_1710 = cpar!("f0_1710", 50.0, 50.0);

    let f0 = gluex::FrozenKMatrix::new(
        "f0",
        2,
        KMatrixConstants {
            g: array![
                [0.74987, 0.06401, -0.23417, 0.0157, -0.14242],
                [-0.01257, 0.00204, -0.01032, 0.267, 0.2278],
                [0.02736, 0.77413, 0.72283, 0.09214, 0.15981],
                [-0.15102, 0.50999, 0.11934, 0.02742, 0.16272],
                [0.36103, 0.13112, 0.36792, -0.04025, -0.17397]
            ],
            m: array![0.51461, 0.90630, 1.23089, 1.46104, 1.69611],
            c: array![
                [0.03728, 0.00000, -0.01398, -0.02203, 0.01397],
                [0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
                [-0.01398, 0.00000, 0.02349, 0.03101, -0.04003],
                [-0.02203, 0.00000, 0.03101, -0.13769, -0.06722],
                [0.01397, 0.00000, -0.04003, -0.06722, -0.28401],
            ],
            m1: array![0.13498, 0.26995, 0.49368, 0.54786, 0.54786],
            m2: array![0.13498, 0.26995, 0.49761, 0.54786, 0.95778],
        },
        0,
        particles.clone(),
        Some(gluex::AdlerZero {
            s_0: 0.009_112_5,
            s_norm: 1.0,
        }),
    )
    .with(pars!(f0_500, f0_980, f0_1370, f0_1500, f0_1710));

    let f2_1270 = cpar!("f2_1270", 50.0, 50.0);
    let f2_1525 = cpar!("f2_1525", 50.0, 50.0);
    let f2_1810 = cpar!("f2_1810", 50.0, 50.0);
    let f2_1950 = cpar!("f2_1950", 50.0, 50.0);

    let f2 = gluex::FrozenKMatrix::new(
        "f2",
        2,
        KMatrixConstants {
            g: array![
                [0.40033, 0.0182, -0.06709, -0.49924],
                [0.15479, 0.173, 0.22941, 0.19295],
                [-0.089, 0.32393, -0.43133, 0.27975],
                [-0.00113, 0.15256, 0.23721, -0.03987]
            ],
            m: array![1.15299, 1.48359, 1.72923, 1.96700],
            c: array![
                [-0.04319, 0.00000, 0.00984, 0.01028],
                [0.00000, 0.00000, 0.00000, 0.00000],
                [0.00984, 0.00000, -0.07344, 0.05533],
                [0.01028, 0.00000, 0.05533, -0.05183],
            ],
            m1: array![0.13498, 0.26995, 0.49368, 0.54786],
            m2: array![0.13498, 0.26995, 0.49761, 0.54786],
        },
        2,
        particles.clone(),
        None,
    )
    .with(pars!(f2_1270, f2_1525, f2_1810, f2_1950));

    let a0_980 = cpar!("a0_980", 50.0, 50.0);
    let a0_1450 = cpar!("a0_1450", 50.0, 50.0);

    let a0 = gluex::FrozenKMatrix::new(
        "a0",
        1,
        KMatrixConstants {
            g: array![[0.43215, 0.19], [-0.28825, 0.43372]],
            m: array![0.95395, 1.26767],
            c: array![[0.00000, 0.00000], [0.00000, 0.00000],],
            m1: array![0.13498, 0.49368],
            m2: array![0.54786, 0.49761],
        },
        0,
        particles.clone(),
        None,
    )
    .with(pars!(a0_980, a0_1450));

    let a2_1320 = cpar!("a2_1320", 50.0, 50.0);
    let a2_1700 = cpar!("a2_1700", 50.0, 50.0);

    let a2 = gluex::FrozenKMatrix::new(
        "a2",
        0,
        KMatrixConstants {
            g: array![[0.30073, 0.68567], [0.21426, 0.12543], [-0.09162, 0.00184]],
            m: array![1.30080, 1.75351],
            c: array![
                [-0.40184, 0.00033, -0.08707],
                [0.00033, -0.21416, -0.06193],
                [-0.08707, -0.06193, -0.17435],
            ],
            m1: array![0.13498, 0.49368, 0.13498],
            m2: array![0.54786, 0.49761, 0.95778],
        },
        2,
        particles.clone(),
        None,
    )
    .with(pars!(a2_1320, a2_1700));

    let amp: Amplitude = ((&f0 + &a0) * zlm_00p.real() + (&f2 + &a2) * zlm_22p.real()).norm_sqr()
        + ((&f0 + &a0) * zlm_00p.real() + (&f2 + &a2) * zlm_22p.real()).norm_sqr();

    let mut dataset = gluex::open_gluex("data.parquet", true);

    println!("Resolving...");
    let before = Instant::now();
    amp.par_resolve_dependencies(&mut dataset);
    println!("{:.2?}", before.elapsed());
    println!("par");
    for _ in 0..10 {
        let before = Instant::now();
        let y = amp.par_evaluate_on(
            &[
                f0_500, f0_980, f0_1370, f0_1500, f0_1710, f2_1270, f2_1525, f2_1810, f2_1950,
                a0_980, a0_1450, a2_1320, a2_1700,
            ],
            &dataset,
        );
        let sum: Complex64 = y.iter().sum();
        println!("{sum}");
        println!("{:.2?}", before.elapsed());
    }
    println!("{}", f0_500.cscalar().value());
}
