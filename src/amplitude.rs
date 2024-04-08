use std::{
    cmp::Ordering,
    fmt::Debug,
    sync::{Arc, RwLock},
    time::Instant,
};

use gomez::Function;
use gomez::Problem;
use num_complex::Complex64;
use rayon::prelude::*;

use crate::dataset::{Dataset, Event};
use indexmap::IndexMap as OHashMap;

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
/// assert_eq!(amplitude!("MyAmplitude", A).read().unwrap().compute(&[], &Event::default()), Complex64::new(0.0, 0.0));
/// ```
#[macro_export]
macro_rules! amplitude {
    ($name:expr, $node:expr) => {{
        use std::sync::{Arc, RwLock};
        Arc::new(RwLock::new(Amplitude::new($name, $node)))
    }};
}

/// Creates a wrapped [`Scalar`] which can be registered by a [`Manager`].
///
/// This macro is a convenience method which takes a name and a [`Node`] and generates a new
/// [`Scalar`] wrapped in a [`RwLock`] which is wrapped in an [`Arc`].
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use rustitude::prelude::*;
/// use num_complex::Complex64;
/// assert_eq!(scalar!("MyScalar").read().unwrap().compute(&[4.3], &Event::default()), Complex64::new(4.3, 0.0));
/// ```
#[macro_export]
macro_rules! scalar {
    ($name:expr) => {{
        use std::sync::{Arc, RwLock};
        Arc::new(RwLock::new(Amplitude::scalar($name)))
    }};
}

/// Creates a wrapped [`ComplexScalar`] which can be registered by a [`Manager`].
///
/// This macro is a convenience method which takes a name and a [`Node`] and generates a new
/// [`ComplexScalar`] wrapped in a [`RwLock`] which is wrapped in an [`Arc`].
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use rustitude::prelude::*;
/// use num_complex::Complex64;
/// assert_eq!(cscalar!("MyCScalar").read().unwrap().compute(&[4.3, 6.2], &Event::default()), Complex64::new(4.3, 6.2));
/// ```
#[macro_export]
macro_rules! cscalar {
    ($name:expr) => {{
        use std::sync::{Arc, RwLock};
        Arc::new(RwLock::new(Amplitude::cscalar($name)))
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
pub struct Amplitude {
    /// A name which uniquely identifies an [`Amplitude`] within a sum and group.
    pub name: String,
    /// A [`Node`] which contains all of the operations needed to compute a [`Complex64`] from an
    /// [`Event`] in a [`Dataset`], a [`Vec<f64>`] of parameter values, and possibly some
    /// precomputed values.
    pub node: Box<dyn Node>,
}
impl Debug for Amplitude {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.name)?;
        Ok(())
    }
}
impl Amplitude {
    pub fn new<N: Node + 'static>(name: &str, node: N) -> Self {
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
            node: Box::new(node),
        }
    }
    pub fn scalar(name: &str) -> Self {
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
        Self {
            name: name.to_string(),
            node: Box::new(Scalar),
        }
    }
    pub fn cscalar(name: &str) -> Self {
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
        Self {
            name: name.to_string(),
            node: Box::new(ComplexScalar),
        }
    }
    pub fn precompute(&mut self, dataset: &Dataset) {
        //! Precalculates the stored [`Node`].
        //!
        //! This method is automatically called when a new [`Amplitude`] is registered by a
        //! [`Manager`]
        //!
        //! See also: [`Manager::register`], [`Node::precalculate`]
        self.node.precalculate(dataset);
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
        self.node.calculate(parameters, event)
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

/// An enum which specifies if a given parameter is fixed or free and gives the corresponding value
/// and input index.
///
/// This enum is mostly used internally by the [`Manager`] struct. It contains two types which both
/// hold tuples with a [`usize`] as the first member. This [`usize`] corresponds to the index this
/// parameter should be sourced from in the input vector. [`ParameterType::Fixed`] parameters will
/// have input indices larger than the number of free parameters in the system, and parameter
/// indices will automatically be incremented and decremented when new parameters are added, fixed,
/// or constrained.
///
/// See also: [`Manager::fix`], [`Manager::free`], [`Manager::constrain`]
#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
pub enum ParameterType {
    Free(usize),
    Fixed(usize, f64),
}
impl ParameterType {
    pub fn increment(&mut self) {
        //! Increments the index by `1`.
        match self {
            Self::Free(ref mut ind) => *ind += 1,
            Self::Fixed(ref mut ind, _) => *ind += 1,
        }
    }
    pub fn decrement(&mut self) {
        //! Decrements the index by `1`.
        match self {
            Self::Free(ref mut ind) => *ind -= 1,
            Self::Fixed(ref mut ind, _) => *ind -= 1,
        }
    }
    pub fn set_index(&mut self, index: usize) {
        //! Sets the index to a given value.
        match self {
            Self::Free(ref mut ind) => *ind = index,
            Self::Fixed(ref mut ind, _) => *ind = index,
        }
    }
    pub fn get_index(&self) -> usize {
        //! Getter method for the index.
        match self {
            Self::Free(ref ind) => *ind,
            Self::Fixed(ref ind, _) => *ind,
        }
    }
    pub fn fix(&mut self, index: usize, value: f64) {
        //! Converts a [`ParameterType::Free`] to [`ParameterType::Fixed`] with a given index and
        //! value.
        if let Self::Free(_) = self {
            *self = Self::Fixed(index, value);
        }
    }
    pub fn free(&mut self, index: usize) {
        //! Converts a [`ParameterType::Fixed`] to [`ParameterType::Free`].
        if let Self::Fixed(_, _) = self {
            *self = Self::Free(index);
        }
    }
}

#[derive(Debug)]
pub enum AmplitudeType {
    Activated(Arc<RwLock<Amplitude>>),
    Deactivated(Arc<RwLock<Amplitude>>),
}
impl AmplitudeType {
    pub fn is_activated(&self) -> bool {
        match self {
            Self::Activated(_) => true,
            Self::Deactivated(_) => false,
        }
    }
    pub fn activate(&mut self) {
        if let Self::Deactivated(ref arc) = self {
            *self = Self::Activated(Arc::clone(arc));
        }
    }
    pub fn deactivate(&mut self) {
        if let Self::Activated(ref arc) = self {
            *self = Self::Deactivated(Arc::clone(arc));
        }
    }
    pub fn get_amplitude(&self) -> &Arc<RwLock<Amplitude>> {
        match self {
            Self::Activated(ref arc) => arc,
            Self::Deactivated(ref arc) => arc,
        }
    }
}

type SumMap = OHashMap<String, OHashMap<String, Vec<AmplitudeType>>>;
type ParMap = OHashMap<String, OHashMap<String, OHashMap<String, Vec<(String, ParameterType)>>>>;

#[derive(Debug)]
pub struct Manager<'d> {
    pub sums: SumMap,
    pub pars: ParMap,
    data: &'d Dataset,
    variable_count: usize,
    fixed_variable_count: usize,
}

impl<'d> Manager<'d> {
    pub fn new(dataset: &'d Dataset) -> Self {
        Self {
            sums: SumMap::default(),
            pars: ParMap::default(),
            data: dataset,
            variable_count: 0,
            fixed_variable_count: 0,
        }
    }
    fn get_parametertype(
        &self,
        sum_name: &str,
        group_name: &str,
        amplitude_name: &str,
        parameter_name: &str,
    ) -> ParameterType {
        self.pars
            .get(sum_name)
            .unwrap_or_else(|| panic!("Could not find {}", sum_name))
            .get(group_name)
            .unwrap_or_else(|| panic!("Could not find {}", group_name))
            .get(amplitude_name)
            .unwrap_or_else(|| panic!("Could not find {}", amplitude_name))
            .iter()
            .find(|(par_name, _)| *par_name == parameter_name)
            .map(|(_, index)| *index)
            .unwrap_or_else(|| panic!("Could not find {}", parameter_name))
    }
    fn get_amplitudetype(
        &self,
        sum_name: &str,
        group_name: &str,
        amplitude_name: &str,
    ) -> &AmplitudeType {
        self.sums
            .get(sum_name)
            .unwrap_or_else(|| panic!("Could not find {}", sum_name))
            .get(group_name)
            .unwrap_or_else(|| panic!("Could not find {}", group_name))
            .iter()
            .find(|amplitude_type| {
                amplitude_type.get_amplitude().read().unwrap().name == amplitude_name
            })
            .unwrap_or_else(|| panic!("Could not find {}", amplitude_name))
    }
    fn get_amplitudetype_mut(
        &mut self,
        sum_name: &str,
        group_name: &str,
        amplitude_name: &str,
    ) -> &mut AmplitudeType {
        self.sums
            .get_mut(sum_name)
            .unwrap_or_else(|| panic!("Could not find {}", sum_name))
            .get_mut(group_name)
            .unwrap_or_else(|| panic!("Could not find {}", group_name))
            .iter_mut()
            .find(|amplitude_type| {
                amplitude_type.get_amplitude().read().unwrap().name == amplitude_name
            })
            .unwrap_or_else(|| panic!("Could not find {}", amplitude_name))
    }
    pub fn register(
        &mut self,
        sum_name: &str,
        group_name: &str,
        amplitude: &Arc<RwLock<Amplitude>>,
    ) {
        amplitude.write().unwrap().precompute(self.data);
        let amp_name = amplitude.read().unwrap().name.clone();
        self.sums
            .entry(sum_name.to_string())
            .and_modify(|sum_entry| {
                sum_entry
                    .entry(group_name.to_string())
                    .and_modify(|group_entry| {
                        group_entry.push(AmplitudeType::Activated(Arc::clone(amplitude)))
                    })
                    .or_insert(vec![AmplitudeType::Activated(Arc::clone(amplitude))]);
            })
            .or_insert_with(|| {
                let mut sum_map = OHashMap::default();
                sum_map.insert(
                    group_name.to_string(),
                    vec![AmplitudeType::Activated(Arc::clone(amplitude))],
                );
                sum_map
            });
        let mut pars: Vec<(String, ParameterType)> = Vec::new();
        if let Some(parameter_names) = amplitude.read().unwrap().node.parameters() {
            for parameter_name in parameter_names {
                pars.push((
                    parameter_name.clone(),
                    ParameterType::Free(self.variable_count),
                ));
                for (_, sum) in self.pars.iter_mut() {
                    for (_, group) in sum.iter_mut() {
                        for (_, amplitude) in group.iter_mut() {
                            for (_, partype) in amplitude.iter_mut() {
                                if partype.get_index() >= self.variable_count {
                                    partype.increment();
                                }
                            }
                        }
                    }
                }
                self.variable_count += 1;
            }
        }
        self.pars
            .entry(sum_name.to_string())
            .and_modify(|sum_entry| {
                sum_entry
                    .entry(group_name.to_string())
                    .and_modify(|group_entry| {
                        group_entry.insert(amp_name.clone(), pars.clone());
                    })
                    .or_insert_with(|| {
                        let mut group_map = OHashMap::default();
                        group_map.insert(amp_name.clone(), pars.clone());
                        group_map
                    });
            })
            .or_insert_with(|| {
                let mut sum_map = OHashMap::default();
                let mut group_map = OHashMap::default();
                group_map.insert(amp_name.clone(), pars);
                sum_map.insert(group_name.to_string(), group_map);
                sum_map
            });
    }
    pub fn activate(&mut self, amplitude: (&str, &str, &str)) {
        let (sum_name, group_name, amplitude_name) = amplitude;
        self.get_amplitudetype_mut(sum_name, group_name, amplitude_name)
            .activate();
    }
    pub fn deactivate(&mut self, amplitude: (&str, &str, &str)) {
        let (sum_name, group_name, amplitude_name) = amplitude;
        self.get_amplitudetype_mut(sum_name, group_name, amplitude_name)
            .deactivate();
    }
    pub fn fix(&mut self, parameter: (&str, &str, &str, &str), value: f64) {
        let (sum_name, group_name, amplitude_name, parameter_name) = parameter;
        let partype = self.get_parametertype(sum_name, group_name, amplitude_name, parameter_name);
        let new_partype =
            ParameterType::Fixed(self.variable_count + self.fixed_variable_count, value);
        self.apply_to_parameters(|other| {
            if other.get_index() == partype.get_index() {
                *other = new_partype;
            }
            if other.get_index() > partype.get_index() {
                other.decrement();
            }
        });
        self.fixed_variable_count += 1;
        self.variable_count -= 1;
    }
    pub fn free(&mut self, parameter: (&str, &str, &str, &str)) {
        let (sum_name, group_name, amplitude_name, parameter_name) = parameter;
        let partype = self.get_parametertype(sum_name, group_name, amplitude_name, parameter_name);
        let new_partype = ParameterType::Free(self.variable_count);
        self.apply_to_parameters(|other| match other.get_index().cmp(&partype.get_index()) {
            Ordering::Less => {
                if other.get_index() >= new_partype.get_index() {
                    other.increment();
                }
            }
            Ordering::Equal => *other = new_partype,
            Ordering::Greater => {}
        });
        self.fixed_variable_count -= 1;
        self.variable_count += 1;
    }
    fn apply_to_amplitudes(&mut self, closure: impl Fn(&mut AmplitudeType)) {
        for (_, sum) in self.sums.iter_mut() {
            for (_, group) in sum.iter_mut() {
                for amplitudetype in group.iter_mut() {
                    closure(amplitudetype)
                }
            }
        }
    }
    fn apply_to_parameters(&mut self, closure: impl Fn(&mut ParameterType)) {
        for (_, sum) in self.pars.iter_mut() {
            for (_, group) in sum.iter_mut() {
                for (_, amplitude) in group.iter_mut() {
                    for (_, partype) in amplitude.iter_mut() {
                        closure(partype)
                    }
                }
            }
        }
    }
    pub fn constrain(
        &mut self,
        parameter_1: (&str, &str, &str, &str),
        parameter_2: (&str, &str, &str, &str),
    ) {
        let (sum_name_1, group_name_1, amplitude_name_1, parameter_name_1) = parameter_1;
        let (sum_name_2, group_name_2, amplitude_name_2, parameter_name_2) = parameter_2;
        let partype_1 =
            self.get_parametertype(sum_name_1, group_name_1, amplitude_name_1, parameter_name_1);
        let partype_2 =
            self.get_parametertype(sum_name_2, group_name_2, amplitude_name_2, parameter_name_2);
        let index_1 = partype_1.get_index();
        let index_2 = partype_2.get_index();
        self.apply_to_parameters(|partype| {
            let par_index = partype.get_index();
            if index_1 > index_2 {
                if par_index == index_1 {
                    *partype = partype_2;
                }
                if par_index > index_1 {
                    partype.decrement();
                }
            }
            if index_1 < index_2 {
                if par_index == index_2 {
                    *partype = partype_1;
                }
                if par_index > index_2 {
                    partype.decrement();
                }
            }
        });
        self.variable_count -= 1;
    }
    pub fn constrain_amplitude(
        &mut self,
        group_1: (&str, &str, &str),
        group_2: (&str, &str, &str),
    ) {
        let (sum_name_1, group_name_1, amplitude_name_1) = group_1;
        let (sum_name_2, group_name_2, amplitude_name_2) = group_2;
        // TODO: this will fail if amplitude at 1 and 2 have different parameter names!
        let parameter_names: Vec<String> = self
            .pars
            .get(sum_name_1)
            .unwrap_or_else(|| panic!("Could not find {}", sum_name_1))
            .get(group_name_1)
            .unwrap_or_else(|| panic!("Could not find {}", group_name_1))
            .get(amplitude_name_1)
            .unwrap_or_else(|| panic!("Could not find {}", amplitude_name_1))
            .iter()
            .map(|parameter| parameter.0.clone())
            .collect();
        for parameter_name in parameter_names {
            self.constrain(
                (sum_name_1, group_name_1, amplitude_name_1, &parameter_name),
                (sum_name_2, group_name_2, amplitude_name_2, &parameter_name),
            )
        }
    }
    pub fn precompute(&mut self) {
        for (_, sum) in self.sums.iter_mut() {
            for (_, group) in sum.iter_mut() {
                for amplitude in group.iter_mut() {
                    amplitude
                        .get_amplitude()
                        .write()
                        .unwrap()
                        .precompute(self.data)
                }
            }
        }
    }
    fn _compute(&self, parameters: &[f64], event: &Event) -> f64 {
        self.sums
            .values()
            .zip(self.pars.values())
            .map(|(sum, sum_parameters)| {
                sum.values()
                    .zip(sum_parameters.values())
                    .map(|(group, group_parameters)| {
                        group
                            .iter()
                            .zip(group_parameters.values())
                            .map(|(amplitude_type, amplitude_parameters)| {
                                let amp_parameters: Vec<_> = amplitude_parameters
                                    .iter()
                                    .map(|param| match param.1 {
                                        ParameterType::Free(ind) => parameters[ind],
                                        ParameterType::Fixed(_, val) => val,
                                    })
                                    .collect();
                                if amplitude_type.is_activated() {
                                    amplitude_type
                                        .get_amplitude()
                                        .read()
                                        .unwrap()
                                        .compute(&amp_parameters, event)
                                } else {
                                    0.0.into()
                                }
                            })
                            .product::<Complex64>()
                    })
                    .sum::<Complex64>()
                    .norm_sqr()
            })
            .sum()
    }
    pub fn compute(&self, parameters: &[f64]) -> Vec<f64> {
        self.data
            .par_iter()
            .map(|event| self._compute(parameters, event))
            .collect()
    }
}

impl<'d> Problem for Manager<'d> {
    type Field = f64;

    fn domain(&self) -> gomez::Domain<Self::Field> {
        gomez::Domain::unconstrained(self.variable_count)
    }
}

impl<'d> Function for Manager<'d> {
    fn apply<Sx>(
        &self,
        parameters: &nalgebra::Vector<Self::Field, nalgebra::Dyn, Sx>,
    ) -> Self::Field
    where
        Sx: nalgebra::Storage<Self::Field, nalgebra::Dyn> + nalgebra::IsContiguous,
    {
        self.compute(parameters.as_slice()).iter().sum()
    }
}

pub struct MultiManager<'a> {
    managers: Vec<Manager<'a>>,
}

impl<'a> MultiManager<'a> {
    pub fn new(datasets: Vec<&'a Dataset>) -> Self {
        Self {
            managers: datasets.iter().map(|ds| Manager::new(ds)).collect(),
        }
    }
    pub fn register(
        &mut self,
        sum_name: &str,
        group_name: &str,
        amplitude: &Arc<RwLock<Amplitude>>,
    ) {
        self.managers.iter_mut().for_each(|manager| {
            let amp = (*amplitude).clone(); // TODO: This doesn't actually work.
            manager.register(sum_name, group_name, &amp);
        });
    }
    pub fn activate(&mut self, amplitude: (&str, &str, &str)) {
        self.managers.iter_mut().for_each(|manager| {
            manager.activate(amplitude);
        });
    }
    pub fn deactivate(&mut self, amplitude: (&str, &str, &str)) {
        self.managers.iter_mut().for_each(|manager| {
            manager.deactivate(amplitude);
        });
    }
    pub fn fix(&mut self, parameter: (&str, &str, &str, &str), value: f64) {
        self.managers.iter_mut().for_each(|manager| {
            manager.fix(parameter, value);
        });
    }
    pub fn free(&mut self, parameter: (&str, &str, &str, &str)) {
        self.managers.iter_mut().for_each(|manager| {
            manager.free(parameter);
        });
    }
    pub fn constrain(
        &mut self,
        parameter_1: (&str, &str, &str, &str),
        parameter_2: (&str, &str, &str, &str),
    ) {
        self.managers.iter_mut().for_each(|manager| {
            manager.constrain(parameter_1, parameter_2);
        });
    }
    pub fn constrain_amplitude(
        &mut self,
        group_1: (&str, &str, &str),
        group_2: (&str, &str, &str),
    ) {
        self.managers.iter_mut().for_each(|manager| {
            manager.constrain_amplitude(group_1, group_2);
        });
    }
    pub fn precompute(&mut self) {
        self.managers.iter_mut().for_each(|manager| {
            manager.precompute();
        });
    }
}

pub struct ExtendedLogLikelihood<'a> {
    pub manager: MultiManager<'a>,
}
impl<'a> ExtendedLogLikelihood<'a> {
    pub fn new(data: &'a Dataset, monte_carlo: &'a Dataset) -> Self {
        Self {
            manager: MultiManager::new(vec![data, monte_carlo]),
        }
    }
    pub fn compute(&self, parameters: &[f64]) -> f64 {
        let now = Instant::now();
        let data_result: f64 = self.manager.managers[0]
            .compute(parameters)
            .iter()
            .map(|res| res.ln())
            .sum();
        let mc_result: f64 = self.manager.managers[1].compute(parameters).iter().sum();
        let n_data = self.manager.managers[0].data.len() as f64;
        let n_mc = self.manager.managers[1].data.len() as f64;
        println!("{:?}", now.elapsed());
        let res = data_result - (n_data / n_mc) * mc_result;
        println!("{res}");
        res
    }
}

impl<'d> Problem for ExtendedLogLikelihood<'d> {
    type Field = f64;

    fn domain(&self) -> gomez::Domain<Self::Field> {
        gomez::Domain::unconstrained(self.manager.managers[0].variable_count)
    }
}

impl<'d> Function for ExtendedLogLikelihood<'d> {
    fn apply<Sx>(
        &self,
        parameters: &nalgebra::Vector<Self::Field, nalgebra::Dyn, Sx>,
    ) -> Self::Field
    where
        Sx: nalgebra::Storage<Self::Field, nalgebra::Dyn> + nalgebra::IsContiguous,
    {
        self.compute(parameters.as_slice())
    }
}
