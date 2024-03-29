#![allow(dead_code)]
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![feature(associated_type_bounds)]
pub mod amplitude;
pub mod dataset;
pub mod four_momentum;
pub mod gluex;
// pub mod manager;
pub mod prelude {
    pub use crate::amplitude::{Amplitude, Node};
    pub use crate::dataset::{Dataset, Event};
    pub use crate::four_momentum::FourMomentum;
}
