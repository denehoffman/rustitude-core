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

#[cfg(test)]
pub mod test {
    use crate::prelude::*;

    #[test]
    fn add_four_momenta() {
        let a = FourMomentum::new(1.0, 2.0, 3.0, 4.0);
        let b = FourMomentum::new(10.0, 20.0, 30.0, 40.0);
        assert_eq!(a + b, FourMomentum::new(11.0, 22.0, 33.0, 44.0))
    }
}
