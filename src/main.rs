use rand::prelude::*;
use std::hint::black_box;
use std::sync::Arc;

use rustitude::prelude::*;
use rustitude::{amplitude::Manager, gluex::KMatrixF0};

fn main() {
    let mut rng = rand::thread_rng();
    let ds = Dataset::from_parquet("accmc_pol.parquet", true);
    let mut a_kmatrix_f0 = Amplitude::new("KMatrix[F0]", KMatrixF0::default());
    let mut m = Manager::new(&ds);
    a_kmatrix_f0.precompute(&ds);
    let arca_kmatrix_f0 = Arc::new(a_kmatrix_f0);
    m.register("sum", "F0", &arca_kmatrix_f0);
    (0..100).for_each(|_| {
        let res: Vec<f64> = m.compute(vec![rng.gen(); 10]);
        black_box(res);
    });
}
