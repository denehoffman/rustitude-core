use std::sync::Arc;

use num_complex::Complex64;
use rayon::prelude::*;
use rustitude::amplitude::Manager;
use rustitude::gluex::{Wave, Ylm};
use rustitude::prelude::*;

fn main() {
    let ds = Dataset::from_parquet("data_pol.parquet", true);
    let mut ta = Amplitude::new("S0", Ylm(Wave::S0));
    let mut tb = Amplitude::new("D2", Ylm(Wave::D2));
    let mut tc = Amplitude::new("P1", Ylm(Wave::P1));
    let p1 = Amplitude::cscalar("Param");
    let mut m = Manager::new(&ds);
    ta.precompute(&ds);
    tb.precompute(&ds);
    tc.precompute(&ds);
    let ata = Arc::new(ta);
    let atb = Arc::new(tb);
    let atc = Arc::new(tc);
    let ap1 = Arc::new(p1);
    m.register("sum1", "Term1", &ata);
    m.register("sum1", "Term1", &atb);
    m.register("sum1", "Term1", &ap1);
    m.register("sum1", "Term2", &atc);
    m.register("sum2", "Term3", &ata);
    dbg!(&m.sums);
    let test = m.compute(&vec![
        vec![vec![vec![], vec![], vec![100.0, 100.0]]],
        vec![vec![vec![]]],
    ]);
    dbg!(test[0]);
}
