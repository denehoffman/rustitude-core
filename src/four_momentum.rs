use nalgebra::Vector3;
use ndarray::{array, Array1, Array2};
use std::ops::{Add, Sub};

#[derive(Debug, Clone, PartialEq)]
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
    //! of the form $`(E, \overrightarrow{p})`$ where $E$ is the energy and $`\overrightarrow{p}`$ is the
    //! momentum.
    //!
    //! # Examples
    //! ```
    //! use rustitude::prelude::*;
    //!
    //! let vec_a = FourMomentum::new(1.3, 0.2, 0.3, 0.1);
    //! let vec_b = FourMomentum::new(4.2, 0.5, 0.4, 0.5);
    //! ```

    pub fn new(e: f64, px: f64, py: f64, pz: f64) -> Self {
        //! Create a new [`FourMomentum`] from energy and momentum components.
        //!
        //! Components are listed in the order $` (E, p_x, p_y, p_z) `$
        Self { e, px, py, pz }
    }

    pub fn to_array(&self) -> Array1<f64> {
        //! Turns a [`FourMomentum`] into a [`ndarray::Array1<f64>`]
        //!
        //! # Examples
        //! ```
        //! use rustitude::prelude::*;
        //! use ndarray::array;
        //!
        //! let vec_a = FourMomentum::new(20.0, 1.0, 0.2, -0.1);
        //! assert_eq!(vec_a.to_array(), array![20.0, 1.0, 0.2, -0.1]);
        //! ```
        array![self.e, self.px, self.py, self.pz]
    }

    pub fn momentum(&self) -> Vector3<f64> {
        //! Extract the 3-momentum as a [`nalgebra::Vector3<f64>`]
        //!
        //! # Examples
        //! ```
        //! use rustitude::prelude::*;
        //! use nalgebra::Vector3;
        //!
        //! let vec_a = FourMomentum::new(20.0, 1.0, 0.2, -0.1);
        //! assert_eq(vec_a.momentum(), Vector3::new(1.0, 0.2, -0.1));
        //! ```
        Vector3::new(self.px, self.py, self.pz)
    }

    pub fn m2(&self) -> f64 {
        //! Calculate the invariant $ m^2 $ for this [`FourMomentum`] instance.
        //!
        //! Calculates $` m^2 = E^2 - \overrightarrow{p}^2 `$
        //!
        //! # Examples
        //! ```
        //! use rustitude::prelude::*;
        //!
        //! let vec_a = FourMomentum::new(20.0, 1.0, 0.2, -0.1);
        //! assert_eq(vec_a.m2(), 20.0 * 20.0 - (1.0 * 1.0 + 0.0 * 0.2 + (-0.1) *
        //! (-0.1)));
        //!
        //! ```
        self.e.powi(2) - self.px.powi(2) - self.py.powi(2) - self.pz.powi(2)
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

    pub fn beta3(&self) -> Vector3<f64> {
        //! Construct the 3-vector $\overrightarrow{\beta}$ where
        //!
        //! $` \overrightarrow{\beta} = \frac{\overrightarrow{p}}{E} `$
        self.momentum() / self.e
    }

    pub fn boost_matrix(&self) -> Array2<f64> {
        //! Construct the Lorentz boost matrix $`\mathbf{\Lambda}`$ where
        //!
        //! ```math
        //! \mathbf{\Lambda} = \begin{pmatrix}
        //! \gamma & -\gamma \beta_x & -\gamma \beta_y & -\gamma \beta_z \\
        //! -\gamma \beta_x & 1 + (\gamma - 1) \frac{\beta_x^2}{\overrightarrow{\beta}^2} & (g - 1) \frac{\beta_x \beta_y}{\overrightarrow{\beta}^2} & (g - 1) \frac{\beta_x \beta_z}{\overrightarrow{\beta}^2} \\
        //! -\gamma \beta_y & (\gamma - 1) \frac{\beta_y \beta_x}{\overrightarrow{\beta}^2} & 1 + (g - 1) \frac{\beta_y^2}{\overrightarrow{\beta}^2} & (g - 1) \frac{\beta_y \beta_z}{\overrightarrow{\beta}^2} \\
        //! -\gamma \beta_z & (\gamma - 1) \frac{\beta_z \beta_x}{\overrightarrow{\beta}^2} & (g - 1) \frac{\beta_z \beta_y}{\overrightarrow{\beta}^2} & 1 + (g - 1) \frac{\beta_z^2}{\overrightarrow{\beta}^2}
        //! \end{pmatrix}
        //! ```
        //! where
        //! $`\overrightarrow{\beta} = \frac{\overrightarrow{p}}{E}`$ and $`\gamma = \frac{1}{\sqrt{1 - \overrightarrow{\beta}^2}}`$.
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
        //! Boosts an instance of [`FourMomentum`] along the $`\overrightarrow{\beta}`$
        //! vector of another [`FourMomentum`].
        //!
        //! Calculates $`\mathbf{\Lambda} \cdot \mathbf{x}`$
        //!
        //! # Examples
        //! ```
        //! use rustitude::prelude::*;
        //!
        //! vec_a = FourMomentum::new(20.0, 1.0, 1.2, -3.4);
        //! vec_a_COM = vec_a.boost_along(&vec_a);
        //! assert_eq!(vec_a_COM, FourMomentum::default()) // this might need to check
        //!                                                // for closeness instead
        //! ```
        let m_boost = other.boost_matrix();
        m_boost.dot(&self.to_array()).into()
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
