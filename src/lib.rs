#![allow(dead_code)]
#![cfg_attr(feature = "simd", feature(portable_simd))]
pub mod amplitude;
pub mod dataset;
pub mod fit;
pub mod four_momentum;
pub mod manager;
pub mod prelude {
    pub use crate::amplitude::{Amplitude, Node};
    pub use crate::dataset::{Dataset, Event};
    pub use crate::fit::Function;
    pub use crate::four_momentum::FourMomentum;
    pub use crate::manager::{ExtendedLogLikelihood, Manage, Manager, MultiManager};
    pub use crate::{amplitude, cscalar, pcscalar, scalar};
    pub use num_complex::Complex64;
}
