use gomez::OptimizerDriver;
use std::sync::Arc;
use std::sync::RwLock;

use rustitude::{
    amplitude::Manager,
    gluex::{ImZlm, KMatrixF0},
};
use rustitude::{
    gluex::{ReZlm, Reflectivity, Wave},
    prelude::*,
};

fn main() {
    let ds = Dataset::from_parquet("data_pol.parquet", true);
    let rezlm_s0p = amplitude!(
        "Name: Re[Zlm(S0+)]",
        ReZlm::new(Wave::S0, Reflectivity::Positive)
    );
    let imzlm_s0p = amplitude!(
        "Name: Im[Zlm(S0+)]",
        ImZlm::new(Wave::S0, Reflectivity::Positive)
    );
    let rezlm_s0n = amplitude!(
        "Name: Re[Zlm(S0-)]",
        ReZlm::new(Wave::S0, Reflectivity::Negative)
    );
    let imzlm_s0n = amplitude!(
        "Name: Im[Zlm(S0-)]",
        ImZlm::new(Wave::S0, Reflectivity::Negative)
    );
    let rezlm_d2p = amplitude!(
        "Name: Re[Zlm(D2+)]",
        ReZlm::new(Wave::D2, Reflectivity::Positive)
    );
    let imzlm_d2p = amplitude!(
        "Name: Im[Zlm(D2+)]",
        ImZlm::new(Wave::D2, Reflectivity::Positive)
    );
    let kmatrix_f0 = amplitude!("Name: F0 K-Matrix", KMatrixF0::default());

    let mut m = Manager::new(&ds);
    m.register("Sum: PosRe", "Group: S0+", &rezlm_s0p);
    m.register("Sum: PosIm", "Group: S0+", &imzlm_s0p);
    m.register("Sum: NegRe", "Group: S0-", &rezlm_s0n);
    m.register("Sum: NegIm", "Group: S0-", &imzlm_s0n);
    m.register("Sum: PosRe", "Group: D2+", &rezlm_d2p);
    m.register("Sum: PosIm", "Group: D2+", &imzlm_d2p);
    m.register("Sum: PosRe", "Group: S0+", &kmatrix_f0);
    m.register("Sum: PosIm", "Group: S0+", &kmatrix_f0);
    m.precompute();
    m.constrain_amplitude(
        ("Sum: PosRe", "Group: S0+", "Name: Re[Zlm(S0+)]"),
        ("Sum: PosIm", "Group: S0+", "Name: Im[Zlm(S0+)]"),
    );
    m.constrain_amplitude(
        ("Sum: NegRe", "Group: S0-", "Name: Re[Zlm(S0-)]"),
        ("Sum: NegIm", "Group: S0-", "Name: Im[Zlm(S0-)]"),
    );
    m.constrain_amplitude(
        ("Sum: PosRe", "Group: D2+", "Name: Re[Zlm(D2+)]"),
        ("Sum: PosIm", "Group: D2+", "Name: Im[Zlm(D2+)]"),
    );
    m.constrain_amplitude(
        ("Sum: PosRe", "Group: S0+", "Name: F0 K-Matrix"),
        ("Sum: PosIm", "Group: S0+", "Name: F0 K-Matrix"),
    );
    m.fix(
        ("Sum: PosRe", "Group: S0+", "Name: F0 K-Matrix", "f0_500 re"),
        0.0,
    );
    m.fix(
        ("Sum: PosRe", "Group: S0+", "Name: F0 K-Matrix", "f0_500 im"),
        0.0,
    );
    m.fix(
        ("Sum: PosRe", "Group: S0+", "Name: F0 K-Matrix", "f0_980 im"),
        0.0,
    );
    m.free(("Sum: PosRe", "Group: S0+", "Name: F0 K-Matrix", "f0_980 im"));
    m.fix(
        ("Sum: PosRe", "Group: S0+", "Name: F0 K-Matrix", "f0_980 im"),
        0.0,
    );
    let mut optimizer = OptimizerDriver::builder(&m)
        .with_initial(vec![1.0; 7])
        .build();
    let (x, fx) = optimizer
        .find(|state| state.iter() >= 100)
        .expect("too many function calls");
    println!("f(x) = {fx}\tx = {x:?}");
}
