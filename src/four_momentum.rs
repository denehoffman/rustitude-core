use nalgebra::{Matrix4, Vector3, Vector4};
use pyo3::prelude::*;
use std::{
    fmt::Display,
    ops::{Add, Sub},
};

#[cfg(feature = "simd")]
use std::simd::prelude::*;

#[cfg(not(feature = "simd"))]
#[pyclass]
#[derive(Debug, Clone, PartialEq, Copy, Default)]
pub struct FourMomentum([f64; 4]);

#[cfg(feature = "simd")]
#[pyclass]
#[derive(Debug, Clone, PartialEq, Copy, Default)]
pub struct FourMomentum(f64x4);

impl Eq for FourMomentum {}

impl Display for FourMomentum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}, ({}, {}, {})]",
            self.e(),
            self.px(),
            self.py(),
            self.pz(),
        )
    }
}

#[pymethods]
impl FourMomentum {
    //! A four-momentum structure with helpful methods for boosts.
    //!
    //! This is the basic structure of a Lorentz four-vector
    //! of the form $`(E, \overrightarrow{p})`$ where $`E`$ is the energy and $`\overrightarrow{p}`$ is the
    //! momentum.
    //!
    //! # Examples
    //! ```
    //! use rustitude_core::prelude::*;
    //!
    //! let vec_a = FourMomentum::new(1.3, 0.2, 0.3, 0.1);
    //! let vec_b = FourMomentum::new(4.2, 0.5, 0.4, 0.5);
    //! ```

    #[cfg(not(feature = "simd"))]
    #[new]
    pub const fn new(e: f64, px: f64, py: f64, pz: f64) -> Self {
        //! Create a new [`FourMomentum`] from energy and momentum components.
        //!
        //! Components are listed in the order $` (E, p_x, p_y, p_z) `$
        Self([e, px, py, pz])
    }

    #[cfg(feature = "simd")]
    #[new]
    pub fn new(e: f64, px: f64, py: f64, pz: f64) -> Self {
        //! Create a new [`FourMomentum`] from energy and momentum components.
        //!
        //! Components are listed in the order $` (E, p_x, p_y, p_z) `$
        Self([e, px, py, pz].into())
    }

    fn __repr__(&self) -> String {
        format!("<FourMomentum ({})>", self)
    }

    fn __str__(&self) -> String {
        self.to_string()
    }

    #[allow(clippy::missing_const_for_fn)]
    pub fn e(&self) -> f64 {
        self.0[0]
    }
    #[allow(clippy::missing_const_for_fn)]
    pub fn px(&self) -> f64 {
        self.0[1]
    }
    #[allow(clippy::missing_const_for_fn)]
    pub fn py(&self) -> f64 {
        self.0[2]
    }
    #[allow(clippy::missing_const_for_fn)]
    pub fn pz(&self) -> f64 {
        self.0[3]
    }

    pub fn set_e(&mut self, value: f64) {
        self.0[0] = value;
    }
    pub fn set_px(&mut self, value: f64) {
        self.0[1] = value;
    }
    pub fn set_py(&mut self, value: f64) {
        self.0[2] = value;
    }
    pub fn set_pz(&mut self, value: f64) {
        self.0[3] = value;
    }

    #[allow(clippy::suboptimal_flops)]
    pub fn m2(&self) -> f64 {
        //! Calculate the invariant $ m^2 $ for this [`FourMomentum`] instance.
        //!
        //! Calculates $` m^2 = E^2 - \overrightarrow{p}^2 `$
        //!
        //! # Examples
        //! ```
        //! use rustitude_core::prelude::*;
        //!
        //! let vec_a = FourMomentum::new(20.0, 1.0, 0.2, -0.1);
        //! //assert_eq!(vec_a.m2(), 20.0 * 20.0 - (1.0 * 1.0 + 0.0 * 0.2 + (-0.1) * (-0.1)));
        //!
        //! ```
        self.e().powi(2) - self.px().powi(2) - self.py().powi(2) - self.pz().powi(2)
    }

    pub fn m(&self) -> f64 {
        //! Calculate the invariant $ m $ for this [`FourMomentum`] instance.
        //!
        //! Calculates $` m = \sqrt{E^2 - \overrightarrow{p}^2} `$
        //!
        //! # See Also:
        //!
        //! [`FourMomentum::m2`]

        self.m2().sqrt()
    }

    pub fn boost_along(&self, other: &Self) -> Self {
        //! Boosts an instance of [`FourMomentum`] along the $`\overrightarrow{\beta}`$
        //! vector of another [`FourMomentum`].
        //!
        //! Calculates $`\mathbf{\Lambda} \cdot \mathbf{x}`$
        //!
        //! # Examples
        //! ```
        //! #[macro_use]
        //! use approx::*;
        //!
        //! use rustitude_core::prelude::*;
        //!
        //! let vec_a = FourMomentum::new(20.0, 1.0, -3.2, 4.0);
        //! let vec_a_COM = vec_a.boost_along(&vec_a);
        //! assert_abs_diff_eq!(vec_a_COM.px(), 0.0, epsilon = 1e-15);
        //! assert_abs_diff_eq!(vec_a_COM.py(), 0.0, epsilon = 1e-15);
        //! assert_abs_diff_eq!(vec_a_COM.pz(), 0.0, epsilon = 1e-15);
        //! ```
        let m_boost = other.boost_matrix();
        (m_boost * Vector4::<f64>::from(self)).into()
    }
}

impl FourMomentum {
    pub fn momentum(&self) -> Vector3<f64> {
        //! Extract the 3-momentum as a [`nalgebra::Vector3<f64>`]
        //!
        //! # Examples
        //! ```
        //! use rustitude_core::prelude::*;
        //! use nalgebra::Vector3;
        //!
        //! let vec_a = FourMomentum::new(20.0, 1.0, 0.2, -0.1);
        //! assert_eq!(vec_a.momentum(), Vector3::new(1.0, 0.2, -0.1));
        //! ```
        Vector3::new(self.px(), self.py(), self.pz())
    }

    pub fn beta3(&self) -> Vector3<f64> {
        //! Construct the 3-vector $\overrightarrow{\beta}$ where
        //!
        //! $` \overrightarrow{\beta} = \frac{\overrightarrow{p}}{E} `$
        self.momentum() / self.e()
    }

    pub fn boost_matrix(&self) -> Matrix4<f64> {
        //! Construct the Lorentz boost matrix $`\mathbf{\Lambda}`$ where
        //!
        //! ```math
        //! \mathbf{\Lambda} = \begin{pmatrix}
        //! \gamma & -\gamma \beta_x & -\gamma \beta_y & -\gamma \beta_z \\
        //! -\gamma \beta_x & 1 + (\gamma - 1) \frac{\beta_x^2}{\overrightarrow{\beta}^2} & (\gamma - 1) \frac{\beta_x \beta_y}{\overrightarrow{\beta}^2} & (\gamma - 1) \frac{\beta_x \beta_z}{\overrightarrow{\beta}^2} \\
        //! -\gamma \beta_y & (\gamma - 1) \frac{\beta_y \beta_x}{\overrightarrow{\beta}^2} & 1 + (\gamma - 1) \frac{\beta_y^2}{\overrightarrow{\beta}^2} & (\gamma - 1) \frac{\beta_y \beta_z}{\overrightarrow{\beta}^2} \\
        //! -\gamma \beta_z & (\gamma - 1) \frac{\beta_z \beta_x}{\overrightarrow{\beta}^2} & (\gamma - 1) \frac{\beta_z \beta_y}{\overrightarrow{\beta}^2} & 1 + (\gamma - 1) \frac{\beta_z^2}{\overrightarrow{\beta}^2}
        //! \end{pmatrix}
        //! ```
        //! where
        //! $`\overrightarrow{\beta} = \frac{\overrightarrow{p}}{E}`$ and $`\gamma = \frac{1}{\sqrt{1 - \overrightarrow{\beta}^2}}`$.
        let b = self.beta3();
        let b2 = b.dot(&b);
        let g = 1.0 / (1.0 - b2).sqrt();
        Matrix4::new(
            g,
            -g * b[0],
            -g * b[1],
            -g * b[2],
            -g * b[0],
            1.0 + (g - 1.0) * b[0] * b[0] / b2,
            (g - 1.0) * b[0] * b[1] / b2,
            (g - 1.0) * b[0] * b[2] / b2,
            -g * b[1],
            (g - 1.0) * b[1] * b[0] / b2,
            1.0 + (g - 1.0) * b[1] * b[1] / b2,
            (g - 1.0) * b[1] * b[2] / b2,
            -g * b[2],
            (g - 1.0) * b[2] * b[0] / b2,
            (g - 1.0) * b[2] * b[1] / b2,
            1.0 + (g - 1.0) * b[2] * b[2] / b2,
        )
    }
}

impl From<FourMomentum> for Vector4<f64> {
    fn from(val: FourMomentum) -> Self {
        Self::new(val.e(), val.px(), val.py(), val.pz())
    }
}

impl From<&FourMomentum> for Vector4<f64> {
    fn from(val: &FourMomentum) -> Self {
        Self::new(val.e(), val.px(), val.py(), val.pz())
    }
}

#[cfg(not(feature = "simd"))]
impl From<Vector4<f64>> for FourMomentum {
    fn from(value: Vector4<f64>) -> Self {
        Self([value[0], value[1], value[2], value[3]])
    }
}

#[cfg(feature = "simd")]
impl From<Vector4<f64>> for FourMomentum {
    fn from(value: Vector4<f64>) -> Self {
        Self(Simd::from_array([value[0], value[1], value[2], value[3]]))
    }
}

#[cfg(not(feature = "simd"))]
impl From<&Vector4<f64>> for FourMomentum {
    fn from(value: &Vector4<f64>) -> Self {
        Self([value[0], value[1], value[2], value[3]])
    }
}

#[cfg(feature = "simd")]
impl From<&Vector4<f64>> for FourMomentum {
    fn from(value: &Vector4<f64>) -> Self {
        Self(Simd::from_array([value[0], value[1], value[2], value[3]]))
    }
}

#[cfg(not(feature = "simd"))]
impl From<Vec<f64>> for FourMomentum {
    fn from(value: Vec<f64>) -> Self {
        Self([value[0], value[1], value[2], value[3]])
    }
}

#[cfg(feature = "simd")]
impl From<Vec<f64>> for FourMomentum {
    fn from(value: Vec<f64>) -> Self {
        Self(Simd::from_array([value[0], value[1], value[2], value[3]]))
    }
}

#[cfg(not(feature = "simd"))]
impl From<&Vec<f64>> for FourMomentum {
    fn from(value: &Vec<f64>) -> Self {
        Self([value[0], value[1], value[2], value[3]])
    }
}

#[cfg(feature = "simd")]
impl From<&Vec<f64>> for FourMomentum {
    fn from(value: &Vec<f64>) -> Self {
        Self(Simd::from_array([value[0], value[1], value[2], value[3]]))
    }
}

#[cfg(feature = "simd")]
impl From<FourMomentum> for f64x4 {
    fn from(value: FourMomentum) -> Self {
        value.0
    }
}

#[cfg(not(feature = "simd"))]
impl Add for FourMomentum {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
        ])
    }
}

#[cfg(feature = "simd")]
impl Add for FourMomentum {
    type Output = FourMomentum;
    fn add(self, rhs: Self) -> Self::Output {
        FourMomentum(self.0 + rhs.0)
    }
}

impl Add for &FourMomentum {
    type Output = <FourMomentum as Add>::Output;
    fn add(self, rhs: &FourMomentum) -> Self::Output {
        FourMomentum::add(*self, *rhs)
    }
}

#[cfg(not(feature = "simd"))]
impl Sub for FourMomentum {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
        ])
    }
}

#[cfg(feature = "simd")]
impl Sub for FourMomentum {
    type Output = FourMomentum;
    fn sub(self, rhs: Self) -> Self::Output {
        FourMomentum(self.0 - rhs.0)
    }
}

impl Sub for &FourMomentum {
    type Output = <FourMomentum as Sub>::Output;
    fn sub(self, rhs: &FourMomentum) -> Self::Output {
        FourMomentum::sub(*self, *rhs)
    }
}

impl<'a> std::iter::Sum<&'a Self> for FourMomentum {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |a, b| a + *b)
    }
}

pub fn pyo3_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FourMomentum>()?;
    Ok(())
}
