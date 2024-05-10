#![warn(clippy::nursery)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![cfg_attr(feature = "simd", feature(portable_simd))]
pub mod amplitude;
pub mod dataset;
pub mod four_momentum;
pub mod manager;
pub mod prelude {
    pub use crate::amplitude;
    pub use crate::amplitude::{cscalar, pcscalar, scalar, Amplitude, Node, Piecewise};
    pub use crate::dataset::{Dataset, Event};
    pub use crate::four_momentum::FourMomentum;
    pub use crate::manager::{ExtendedLogLikelihood, Manage, Manager, MultiManager};
    pub use num_complex::Complex64;
}
