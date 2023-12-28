use nalgebra::Vector3;
use ndarray::{array, Array1, Array2};
use std::ops::{Add, Sub};

#[derive(Debug, Clone)]
pub struct FourMomentum {
    pub e: f64,
    pub px: f64,
    pub py: f64,
    pub pz: f64,
}

impl FourMomentum {
    //! A four-momentum structure with helpful methods for boosts.
    //!
    //! This is the basic structure of a Lorentz four-vector
    //! of the form (E, p⃗) where E is the energy and p⃗ is the
    //! momentum.
    //!
    //! # Examples
    //! ```
    //! use rustitude::prelude::*;
    //!
    //! let vec_a = FourMomentum::new(1.3, 0.2, 0.3, 0.1);
    //! let vec_b = FourMomentum::new(4.2, 0.5, 0.4, 0.5);
    //!
    //!
    //! ```

    pub fn new(e: f64, px: f64, py: f64, pz: f64) -> Self {
        Self { e, px, py, pz }
    }

    pub fn as_array(&self) -> Array1<f64> {
        array![self.e, self.px, self.py, self.pz]
    }

    pub fn momentum(&self) -> Vector3<f64> {
        Vector3::new(self.px, self.py, self.pz)
    }

    pub fn m2(&self) -> f64 {
        self.e.powi(2) - self.px.powi(2) - self.py.powi(2) - self.pz.powi(2)
    }

    pub fn m(&self) -> f64 {
        self.m2().sqrt()
    }

    pub fn beta3(&self) -> Vector3<f64> {
        self.momentum() / self.e
    }

    pub fn boost_matrix(&self) -> Array2<f64> {
        let b = self.beta3();
        let b2 = b.dot(&b);
        let g = 1.0 / (1.0 - b2).sqrt();
        let out = array![
            [g, -g * b[0], -g * b[1], -g * b[2]],
            [
                -g * b[0],
                1.0 + (g - 1.0) * b[0] * b[0] / b2,
                (g - 1.0) * b[0] * b[1] / b2,
                (g - 1.0) * b[0] * b[2] / b2
            ],
            [
                -g * b[1],
                (g - 1.0) * b[1] * b[0] / b2,
                1.0 + (g - 1.0) * b[1] * b[1] / b2,
                (g - 1.0) * b[1] * b[2] / b2
            ],
            [
                -g * b[2],
                (g - 1.0) * b[2] * b[0] / b2,
                (g - 1.0) * b[2] * b[1] / b2,
                1.0 + (g - 1.0) * b[2] * b[2] / b2
            ]
        ];
        out
    }

    pub fn boost_along(&self, other: &Self) -> Self {
        let m_boost = other.boost_matrix();
        m_boost.dot(&self.as_array()).into()
    }
}

impl From<FourMomentum> for Array1<f64> {
    fn from(val: FourMomentum) -> Self {
        array![val.e, val.px, val.py, val.pz]
    }
}

impl From<Array1<f64>> for FourMomentum {
    fn from(value: Array1<f64>) -> Self {
        Self {
            e: value[0],
            px: value[1],
            py: value[2],
            pz: value[3],
        }
    }
}

impl Add for FourMomentum {
    type Output = FourMomentum;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            e: self.e + rhs.e,
            px: self.px + rhs.px,
            py: self.py + rhs.py,
            pz: self.pz + rhs.pz,
        }
    }
}

impl<'a, 'b> Add<&'b FourMomentum> for &'a FourMomentum {
    type Output = FourMomentum;
    fn add(self, rhs: &'b FourMomentum) -> Self::Output {
        FourMomentum {
            e: self.e + rhs.e,
            px: self.px + rhs.px,
            py: self.py + rhs.py,
            pz: self.pz + rhs.pz,
        }
    }
}

impl Sub for FourMomentum {
    type Output = FourMomentum;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            e: self.e - rhs.e,
            px: self.px - rhs.px,
            py: self.py - rhs.py,
            pz: self.pz - rhs.pz,
        }
    }
}

impl<'a, 'b> Sub<&'b FourMomentum> for &'a FourMomentum {
    type Output = FourMomentum;
    fn sub(self, rhs: &'b FourMomentum) -> Self::Output {
        FourMomentum {
            e: self.e - rhs.e,
            px: self.px - rhs.px,
            py: self.py - rhs.py,
            pz: self.pz - rhs.pz,
        }
    }
}

impl Default for FourMomentum {
    fn default() -> Self {
        Self {
            e: 0.0,
            px: 0.0,
            py: 0.0,
            pz: 0.0,
        }
    }
}

impl<'a> std::iter::Sum<&'a FourMomentum> for FourMomentum {
    fn sum<I: Iterator<Item = &'a FourMomentum>>(iter: I) -> Self {
        iter.fold(FourMomentum::default(), |a, b| a + b.clone())
    }
}
