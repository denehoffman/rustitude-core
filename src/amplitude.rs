use num::complex::Complex64;
use parking_lot::RwLock;
use pyo3::prelude::*;
use std::{fmt::Debug, sync::Arc};

use crate::dataset::{Dataset, Event};

/// Creates a wrapped [`Amplitude`] which can be registered by a [`Manager`].
///
/// This macro is a convenience method which takes a name and a [`Node`] and generates a new
/// [`Amplitude`] wrapped in a [`RwLock`] which is wrapped in an [`Arc`].
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use rustitude::prelude::*;
/// use num_complex::Complex64;
/// struct A;
/// impl Node for A {
///     fn precalculate(&mut self, dataset: &Dataset) {}
///     fn calculate(&self, parameters: &[f64], event: &Event) -> Complex64 { 0.0.into() }
///     fn parameters(&self) -> Option<Vec<String>> {None}
/// }
///
/// assert_eq!(amplitude!("MyAmplitude", A).read().compute(&[], &Event::default()), Complex64::new(0.0, 0.0));
/// ```
#[macro_export]
macro_rules! amplitude {
    ($name:expr, $node:expr) => {{
        Amplitude::new($name, Box::new($node))
    }};
}

/// A trait which contains all the required methods for a functioning [`Amplitude`].
///
/// The [`Node`] trait represents any mathematical structure which takes in some parameters and some
/// [`Event`] data and computes a [`Complex64`] for each [`Event`]. This is the fundamental
/// building block of all analyses built with Rustitude. Nodes are intended to be optimized at the
/// user level, so they should be implemented on structs which can store some precalculated data.
///
/// # Examples:
///
/// A [`Node`] for calculating spherical harmonics:
///
/// ```
/// use rustitude::prelude::*;
///
/// use nalgebra::{SMatrix, SVector};
/// use num_complex::Complex64;
/// use rayon::prelude::*;
/// use sphrs::SHEval;
/// use sphrs::{ComplexSH, Coordinates};
///
/// #[derive(Clone, Copy, Default)]
/// #[rustfmt::skip]
/// enum Wave {
///     #[default]
///     S,
///     S0,
///     Pn1, P0, P1, P,
///     Dn2, Dn1, D0, D1, D2, D,
///     Fn3, Fn2, Fn1, F0, F1, F2, F3, F,
/// }
///
/// #[rustfmt::skip]
/// impl Wave {
///     fn l(&self) -> i64 {
///         match self {
///             Self::S0 | Self::S => 0,
///             Self::Pn1 | Self::P0 | Self::P1 | Self::P => 1,
///             Self::Dn2 | Self::Dn1 | Self::D0 | Self::D1 | Self::D2 | Self::D => 2,
///             Self::Fn3 | Self::Fn2 | Self::Fn1 | Self::F0 | Self::F1 | Self::F2 | Self::F3 | Self::F => 3,
///         }
///     }
///     fn m(&self) -> i64 {
///         match self {
///             Self::S | Self::P | Self::D | Self::F => 0,
///             Self::S0 | Self::P0 | Self::D0 | Self::F0 => 0,
///             Self::Pn1 | Self::Dn1 | Self::Fn1 => -1,
///             Self::P1 | Self::D1 | Self::F1 => 1,
///             Self::Dn2 | Self::Fn2 => -2,
///             Self::D2 | Self::F2 => 2,
///             Self::Fn3 => -3,
///             Self::F3 => 3,
///         }
///     }
/// }
///
/// struct Ylm(Wave, Vec<Complex64>);
/// impl Ylm {
///     fn new(wave: Wave) -> Self {
///         Self(wave, Vec::default())
///     }
/// }
/// impl Node for Ylm {
///     fn parameters(&self) -> Option<Vec<String>> {
///         None
///     }
///
///     fn precalculate(&mut self, dataset: &Dataset) {
///         self.1 = dataset
///             .par_iter()
///             .map(|event| {
///                 let resonance = event.daughter_p4s[0] + event.daughter_p4s[1];
///                 let p1 = event.daughter_p4s[0];
///                 let recoil_res = event.recoil_p4.boost_along(&resonance); // Boost to helicity frame
///                 let p1_res = p1.boost_along(&resonance);
///                 let z = -1.0 * recoil_res.momentum().normalize();
///                 let y = event
///                     .beam_p4
///                     .momentum()
///                     .cross(&(-1.0 * event.recoil_p4.momentum()));
///                 let x = y.cross(&z);
///                 let p1_vec = p1_res.momentum();
///                 let p = Coordinates::cartesian(p1_vec.dot(&x), p1_vec.dot(&y), p1_vec.dot(&z));
///                 ComplexSH::Spherical.eval(self.0.l(), self.0.m(), &p)
///             })
///             .collect();
///     }
///
///     fn calculate(&self, _parameters: &[f64], event: &Event) -> Complex64 {
///         self.1[event.index]
///     }
/// }
/// ```
///
/// A [`Node`] which computes a single complex scalar entirely determined by input parameters:
///
/// ```
/// use rustitude::prelude::*;
/// struct ComplexScalar;
/// impl Node for ComplexScalar {
///     fn calculate(&self, parameters: &[f64], _event: &Event) -> Complex64 {
///         Complex64::new(parameters[0], parameters[1])
///     }
///
///     fn parameters(&self) -> Option<Vec<String>> {
///         Some(vec!["real".to_string(), "imag".to_string()])
///     }
///
///     fn precalculate(&mut self, _dataset: &Dataset) {}
/// }
/// ```
pub trait Node: Sync + Send {
    /// A method that is run once and stores some precalculated values given a [`Dataset`] input.
    ///
    /// This method is intended to run expensive calculations which don't actually depend on the
    /// parameters. For instance, to calculate a spherical harmonic, we don't actually need any
    /// other information than what is contained in the [`Event`], so we can calculate a spherical
    /// harmonic for every event once and then retrieve the data in the [`Node::calculate`] method.
    fn precalculate(&mut self, dataset: &Dataset);

    /// A method which runs every time the amplitude is evaluated and produces a [`Complex64`].
    ///
    /// Because this method is run on every evaluation, it should be as lean as possible.
    /// Additionally, you should avoid [`rayon`]'s parallel loops inside this method since we
    /// already parallelize over the [`Dataset`]. This method expects a single [`Event`] as well as
    /// a slice of [`f64`]s. This slice is guaranteed to have the same length and order as
    /// specified in the [`Node::parameters`] method, or it will be empty if that method returns
    /// [`None`].
    fn calculate(&self, parameters: &[f64], event: &Event) -> Complex64;

    /// A method which specifies the number and order of parameters used by the [`Node`].
    ///
    /// This method tells the [`Manager`] how to assign its input [`Vec`] of parameter values to
    /// each [`Node`]. If this method returns [`None`], it is implied that the [`Node`] takes no
    /// parameters as input. Otherwise, the parameter names should be listed in the same order they
    /// are expected to be given as input to the [`Node::calculate`] method.
    fn parameters(&self) -> Option<Vec<String>>;
}

/// A struct which stores a named [`Node`].
///
/// The [`Amplitude`] struct turns a [`Node`] trait into a concrete type and also stores a name
/// associated with the [`Node`]. This allows us to distinguish multiple uses of the same [`Node`]
/// in an analysis, and makes each [`Node`]'s parameters unique.
///
/// The common construction pattern is through the macros [`amplitude!`], [`scalar!`], and
/// [`cscalar`] which create a [`Arc<RwLock<Amplitude>>`], an [`Arc<RwLock<Scalar>>`], and an
/// [`Arc<RwLock<ComplexScalar>>`] respectively.
#[pyclass]
#[derive(Clone)]
pub struct Amplitude {
    /// A name which uniquely identifies an [`Amplitude`] within a sum and group.
    pub name: String,
    /// A [`Node`] which contains all of the operations needed to compute a [`Complex64`] from an
    /// [`Event`] in a [`Dataset`], a [`Vec<f64>`] of parameter values, and possibly some
    /// precomputed values.
    pub node: Arc<RwLock<Box<dyn Node>>>,
}
impl Debug for Amplitude {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.name)?;
        Ok(())
    }
}
impl Amplitude {
    pub fn new(name: &str, node: Box<dyn Node>) -> Self {
        //! Creates an named [`Amplitude`] from a [`Node`].
        //!
        //! The [`amplitude!`] macro is probably the cleaner way of doing this, since it also wraps
        //! this [`Amplitude`] in an [`Arc<RwLock<Amplitude>>`] container which can then be registered by
        //! a [`Manager`].
        //!
        //! # Examples
        //!
        //! Basic usage:
        //!
        //! ```
        //! use rustitude::prelude::*;
        //! use num_complex::Complex64;
        //! struct A;
        //! impl Node for A {
        //!     fn precalculate(&mut self, dataset: &Dataset) {}
        //!     fn calculate(&self, parameters: &[f64], event: &Event) -> Complex64 { 0.0.into() }
        //!     fn parameters(&self) -> Option<Vec<String>> {None}
        //! }
        //!
        //! assert_eq!(Amplitude::new("A", A).name, "A".to_string());
        //! ```
        Self {
            name: name.to_string(),
            node: Arc::new(RwLock::new(node)),
        }
    }
    pub fn precompute(&self, dataset: &Dataset) {
        //! Precalculates the stored [`Node`].
        //!
        //! This method is automatically called when a new [`Amplitude`] is registered by a
        //! [`Manager`]
        //!
        //! See also: [`Manager::register`], [`Node::precalculate`]
        self.node.write().precalculate(dataset);
    }
    pub fn compute(&self, parameters: &[f64], event: &Event) -> Complex64 {
        //! Calculates the stored [`Node`].
        //!
        //! This method is intended to be called by a [`Manager`] in the [`Manager::compute`]
        //! method. You can also use this method to test amplitudes, since the [`Manager::compute`]
        //! method will automatically calculate the absolute-square of the amplitude and return a
        //! [`f64`] rather than a [`Complex64`].
        //!
        //! See also: [`Manager::compute`], [`Node::calculate`]
        self.node.read().calculate(parameters, event)
    }
}

#[pyfunction]
pub fn scalar(name: &str) -> Amplitude {
    //! Creates a named [`Scalar`].
    //!
    //! This is a convenience method to generate an [`Amplitude`] which is just a single free
    //! parameter called `value`. The macro [`scalar!`] will wrap this [`Amplitude`] in an
    //! [`Arc<RwLock<Scalar>>`]> container which can then be registered by a [`Manager`].
    //!
    //! # Examples
    //!
    //! Basic usage:
    //!
    //! ```
    //! use rustitude::prelude::*;
    //! let my_scalar = Amplitude::scalar("MyScalar");
    //! assert_eq!(my_scalar.node.parameters(), Some(vec!["value".to_string()]));
    //! ```
    Amplitude {
        name: name.to_string(),
        node: Arc::new(RwLock::new(Box::new(Scalar))),
    }
}
#[pyfunction]
pub fn cscalar(name: &str) -> Amplitude {
    //! Creates a named [`ComplexScalar`].
    //!
    //! This is a convenience method to generate an [`Amplitude`] which represents a complex
    //! value determined by two parameters, `real` and `imag`. The macro [`cscalar!`] will
    //! wrap this [`Amplitude`] in an [`Arc<RwLock<ComplexScalar>>`]> container which can
    //! then be registered by a [`Manager`].
    //!
    //! # Examples
    //!
    //! Basic usage:
    //!
    //! ```
    //! use rustitude::prelude::*;
    //! let my_cscalar = Amplitude::cscalar("MyComplexScalar");
    //! assert_eq!(my_cscalar.node.parameters(), Some(vec!["real".to_string(), "imag".to_string()]));
    //! ```
    Amplitude {
        name: name.to_string(),
        node: Arc::new(RwLock::new(Box::new(ComplexScalar))),
    }
}
#[pyfunction]
pub fn pcscalar(name: &str) -> Amplitude {
    //! Creates a named [`PolarComplexScalar`].
    //!
    //! This is a convenience method to generate an [`Amplitude`] which represents a complex
    //! value determined by two parameters, `real` and `imag`. The macro [`pcscalar!`] will
    //! wrap this [`Amplitude`] in an [`Arc<RwLock<ComplexScalar>>`]> container which can
    //! then be registered by a [`Manager`].
    //!
    //! # Examples
    //!
    //! Basic usage:
    //!
    //! ```
    //! use rustitude::prelude::*;
    //! let my_pcscalar = Amplitude::pcscalar("MyPolarComplexScalar");
    //! assert_eq!(my_pcscalar.node.parameters(), Some(vec!["mag".to_string(), "phi".to_string()]));
    //! ```
    Amplitude {
        name: name.to_string(),
        node: Arc::new(RwLock::new(Box::new(PolarComplexScalar))),
    }
}

/// A [`Node`] for computing a single scalar value from an input parameter.
///
/// This struct implements [`Node`] to generate a single new parameter called `value`.
///
/// # Parameters:
///
/// - `value`: The value of the scalar.
pub struct Scalar;
impl Node for Scalar {
    fn parameters(&self) -> Option<Vec<String>> {
        Some(vec!["value".to_string()])
    }

    fn precalculate(&mut self, _dataset: &Dataset) {}

    fn calculate(&self, parameters: &[f64], _event: &Event) -> Complex64 {
        Complex64::new(parameters[0], 0.0)
    }
}

/// A [`Node`] for computing a single complex value from two input parameters.
///
/// This struct implements [`Node`] to generate a complex value from two input parameters called
/// `real` and `imag`.
///
/// # Parameters:
///
/// - `real`: The real part of the complex scalar.
/// - `imag`: The imaginary part of the complex scalar.
pub struct ComplexScalar;
impl Node for ComplexScalar {
    fn calculate(&self, parameters: &[f64], _event: &Event) -> Complex64 {
        Complex64::new(parameters[0], parameters[1])
    }

    fn parameters(&self) -> Option<Vec<String>> {
        Some(vec!["real".to_string(), "imag".to_string()])
    }

    fn precalculate(&mut self, _dataset: &Dataset) {}
}

/// A [`Node`] for computing a single complex value from two input parameters in polar form.
///
/// This struct implements [`Node`] to generate a complex value from two input parameters called
/// `mag` and `phi`.
///
/// # Parameters:
///
/// - `mag`: The magnitude of the complex scalar.
/// - `phi`: The phase of the complex scalar.
pub struct PolarComplexScalar;
impl Node for PolarComplexScalar {
    fn calculate(&self, parameters: &[f64], _event: &Event) -> Complex64 {
        parameters[0] * Complex64::cis(parameters[1])
    }

    fn parameters(&self) -> Option<Vec<String>> {
        Some(vec!["mag".to_string(), "phi".to_string()])
    }

    fn precalculate(&mut self, _dataset: &Dataset) {}
}

#[pymodule]
pub fn amplitude(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Amplitude>()?;
    m.add_function(wrap_pyfunction!(scalar, m)?)?;
    m.add_function(wrap_pyfunction!(cscalar, m)?)?;
    m.add_function(wrap_pyfunction!(pcscalar, m)?)?;
    Ok(())
}
