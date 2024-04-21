#![allow(dead_code)]
#![cfg_attr(feature = "simd", feature(portable_simd))]
pub mod amplitude;
pub mod dataset;
pub mod four_momentum;
pub mod manager;
pub mod prelude {
    pub use crate::amplitude::{Amplitude, Node};
    pub use crate::dataset::{Dataset, Event};
    pub use crate::four_momentum::FourMomentum;
    pub use crate::manager::{ExtendedLogLikelihood, Manage, Manager, MultiManager};
    pub use crate::{amplitude, cscalar, pcscalar, scalar};
    pub use num_complex::Complex64;
}
use pyo3::{prelude::*, types::PyDict, wrap_pymodule};

#[pyfunction]
fn testing() {
    println!("Testing!");
}

#[pymodule]
fn rustitude(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sys = PyModule::import_bound(py, "sys")?;
    let sys_modules: Bound<'_, PyDict> = sys.getattr("modules")?.downcast_into()?;

    m.add_function(wrap_pyfunction!(testing, m)?)?;
    m.add_wrapped(wrap_pymodule!(four_momentum::four_momentum))?;
    sys_modules.set_item("rustitude.four_momentum", m.getattr("four_momentum")?)?;
    Ok(())
}
