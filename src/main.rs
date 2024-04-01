use std::sync::Arc;

use argmin::core::Executor;
use argmin::solver::brent::BrentOpt;
use num_complex::Complex64;
use rayon::prelude::*;
use rustitude::amplitude::Manager;
use rustitude::gluex::{ImZlm, ReZlm, Reflectivity, Wave, Ylm};
use rustitude::prelude::*;

fn main() {
    let ds = Dataset::from_parquet("data_pol.parquet", true);
    let mut a_rezlm_s0p = Amplitude::new("ReZlm[S0+]", ReZlm(Wave::S0, Reflectivity::Positive));
    let mut a_imzlm_s0p = Amplitude::new("ImZlm[S0+]", ImZlm(Wave::S0, Reflectivity::Positive));
    let mut a_rezlm_s0n = Amplitude::new("ReZlm[S0-]", ReZlm(Wave::S0, Reflectivity::Negative));
    let mut a_imzlm_s0n = Amplitude::new("ImZlm[S0-]", ImZlm(Wave::S0, Reflectivity::Negative));
    let mut a_rezlm_d2p = Amplitude::new("ReZlm[D2+]", ReZlm(Wave::D2, Reflectivity::Positive));
    let mut a_imzlm_d2p = Amplitude::new("ImZlm[D2+]", ImZlm(Wave::D2, Reflectivity::Positive));
    let mut m = Manager::new(&ds);
    a_rezlm_s0p.precompute(&ds);
    a_imzlm_s0p.precompute(&ds);
    a_rezlm_s0n.precompute(&ds);
    a_imzlm_s0n.precompute(&ds);
    a_rezlm_d2p.precompute(&ds);
    a_imzlm_d2p.precompute(&ds);
    let arca_rezlm_s0p = Arc::new(a_rezlm_s0p);
    let arca_imzlm_s0p = Arc::new(a_imzlm_s0p);
    let arca_rezlm_s0n = Arc::new(a_rezlm_s0n);
    let arca_imzlm_s0n = Arc::new(a_imzlm_s0n);
    let arca_rezlm_d2p = Arc::new(a_rezlm_d2p);
    let arca_imzlm_d2p = Arc::new(a_imzlm_d2p);
    m.register("pos_re", "Re[S0+]", &arca_rezlm_s0p);
    m.register("pos_re", "Re[D2+]", &arca_rezlm_d2p);
    m.register("pos_im", "Re[S0+]", &arca_imzlm_s0p);
    m.register("pos_im", "Re[D2+]", &arca_imzlm_d2p);
    m.register("neg_re", "Re[S0-]", &arca_rezlm_s0n);
    m.register("neg_im", "Re[S0-]", &arca_imzlm_s0n);
    dbg!(&m.pars);
    let test = m.compute(vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ]);
    // dbg!(test);
    //
    //     let solver = BrentOpt::new(
    //         &vec![
    //             vec![vec![vec![], vec![], vec![-100.0, -100.0]]],
    //             vec![vec![vec![]]],
    //         ],
    //         &vec![
    //             vec![vec![vec![], vec![], vec![100.0, 100.0]]],
    //             vec![vec![vec![]]],
    //         ],
    //     );
    //     let res = Executor::new(m, solver)
    //         .configure(|state| state.max_iters(100))
    //         .run()
    //         .unwrap();
}
