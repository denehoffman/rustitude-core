// use derive_new::new;
use parking_lot::RwLock;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::{error::Error, sync::Arc};
use variantly::Variantly;

use num_complex::Complex64;

use crate::prelude::{Dataset, FieldType};

#[macro_export]
macro_rules! par {
    ($name:expr, $value:expr) => {
        Parameter::new($name, ParameterValue::Scalar($value))
    };
}

#[macro_export]
macro_rules! cpar {
    ($name:expr, $value_re:expr, $value_im:expr) => {
        Parameter::new(
            $name,
            ParameterValue::CScalar(Complex64::new($value_re, $value_im)),
        )
    };
}

#[macro_export]
macro_rules! pars {
    ($($item:expr),*) => {
        &[$($item.clone().into()),*]
    };

}

#[macro_export]
macro_rules! var {
    ($struct_type:ident {$($field:ident : $value:expr),* $(,)?}) => {
        {$struct_type {
            $($field: $value),*
        }}
    };
    ($struct_type:path {$($field:ident : $value:expr),* $(,)?}) => {
        {$struct_type {
            $($field: $value),*
        }}
    };
}

#[macro_export]
macro_rules! vars {
    ($($var:ident {$($field:ident : $value:expr),* $(,)?}),* $(,)?) => {
        vec![
            $(
                var!($var {$($field: $value),*})
            ),*
        ]
    };
    ($($var:path {$($field:ident : $value:expr),* $(,)?}),* $(,)?) => {
        vec![
            $(
                var!($var {$($field: $value),*})
            ),*
        ]
    };
}

#[derive(Clone)]
enum Operation<'a> {
    Add(Amplitude<'a>, Amplitude<'a>),
    Sub(Amplitude<'a>, Amplitude<'a>),
    Mul(Amplitude<'a>, Amplitude<'a>),
    Div(Amplitude<'a>, Amplitude<'a>),
    Neg(Amplitude<'a>),
    Sqrt(Amplitude<'a>),
    NormSquare(Amplitude<'a>),
    Real(Amplitude<'a>),
    Imag(Amplitude<'a>),
}

pub type ParMap<'a> = HashMap<String, Parameter<'a>>;
pub type VarMap = HashMap<String, FieldType>;
pub type SendableAmpFn =
    dyn Fn(&ParMap, &VarMap) -> Result<Complex64, Box<dyn Error + Send + Sync>> + Send + Sync;
pub type ArcAmpFn = Arc<RwLock<SendableAmpFn>>;
pub type SendableVarFn = dyn Fn(&VarMap) -> FieldType + Send + Sync;
pub type ArcVarFn = Arc<RwLock<SendableVarFn>>;

#[derive(Default, Clone)]
pub struct Amplitude<'a> {
    pub name: Arc<String>,
    function: Option<ArcAmpFn>,
    internal_parameters: Arc<RwLock<Vec<String>>>,
    pub external_parameters: Arc<RwLock<HashMap<String, Parameter<'a>>>>,
    parameter_mappings: Arc<RwLock<HashMap<Parameter<'a>, String>>>,
    op: Option<Arc<RwLock<Operation<'a>>>>,
    dependencies: Option<Vec<Variable>>,
}

impl<'a> std::fmt::Debug for Amplitude<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} ->\n\tpars = {:?}\n\tmap = {:?}\n\tdeps = {:?}",
            self.name, self.internal_parameters, self.parameter_mappings, self.dependencies
        )
    }
}

impl<'a> Amplitude<'a> {
    //! The [`Amplitude`] struct is at the core of this package. It holds a function which takes
    //! [`ParMap`] and a [`VarMap`] and returns a [`Result`] containing a [`Complex64`].
    //!
    //! Amplitudes can optionally have `internal_parameters` and `dependencies`, which correspond
    //! to [`Vec`]s of [`String`] or [`Variable`] respectively. Amplitudes can depend on
    //! [`Variable`]s, but not vice-versa, and not with other Amplitudes. Amplitudes hold a set of
    //! internal parameter names which amplitude developers can refer to when writing out the
    //! mathematics behind the evaluating function.
    //!
    //! # Examples
    //!
    //! ```
    //! use num_complex::Complex64;
    //! use rustitude::prelude::*;
    //!
    //! let amp = Amplitude::new("MyAmp", |pars: &ParMap, vars: &VarMap| {Ok(pars["parameter"].value.cscalar().unwrap() * 10.0)}, Some(vec!["parameter"]), None);
    //! let mut p = cpar!("MyPar", 2.0, 3.0);
    //! amp.assign(&p, "parameter");
    //!
    //! let mut d: Dataset = Dataset::new(10);
    //! let res: Vec<Complex64> = amp.par_evaluate_on(&d);
    //! assert_eq!(res[0], Complex64::new(20.0, 30.0));
    //! ```
    //!

    pub fn new<F>(
        name: &str,
        function: F,
        internal_parameters: Option<Vec<&str>>,
        dependencies: Option<Vec<Variable>>,
    ) -> Self
    where
        F: 'static
            + Fn(&ParMap, &VarMap) -> Result<Complex64, Box<dyn Error + Send + Sync>>
            + Sync
            + Send,
    {
        let internal_parameters: Arc<RwLock<Vec<String>>> = match internal_parameters {
            Some(pars) => Arc::new(RwLock::new(
                pars.iter().map(std::string::ToString::to_string).collect(),
            )),
            None => Arc::new(RwLock::new(Vec::default())),
        };
        Amplitude {
            name: Arc::new(name.to_string()),
            function: Some(Arc::new(RwLock::new(function))),
            dependencies,
            internal_parameters,
            ..Default::default()
        }
    }

    /// Assign an external parameter to an internal name.
    ///
    /// # Panics
    ///
    /// This function panics if the internal name is not in the list of internal names provided
    /// by the amplitude.
    pub fn assign(&self, external_par: &Parameter<'a>, internal_name: &str) {
        let internal_pars = self.internal_parameters.read();
        if internal_pars.contains(&internal_name.to_string()) {
            self.parameter_mappings
                .write()
                .insert(*external_par, internal_name.to_string());
            self.external_parameters
                .write()
                .insert(external_par.name.to_string(), *external_par);
        } else {
            panic!("Name not found!");
        }
    }

    fn _evaluate(
        &self,
        pars: &ParMap,
        vars: &VarMap,
    ) -> Result<Complex64, Box<dyn Error + Send + Sync>> {
        let internal_pars = self
            .parameter_mappings
            .read()
            .iter()
            .filter_map(|(external, internal)| {
                pars.get(external.name).map(|par| (internal.clone(), *par))
            })
            .collect();
        if let Some(ref func_arc) = self.function {
            let func_rwlock = func_arc.read();
            func_rwlock(&internal_pars, vars)
        } else {
            Err("Function is not set".into())
        }
    }

    /// Evaluate an amplitude for a set of parameters `pars` and a set of variables `vars`.
    ///
    /// # Errors
    ///
    /// Returns an error if anything happens to raise an error in the evaluation of any
    /// sub-amplitudes or in running _evaluate, which will either pass errors from the internal
    /// amplitude's function, or will raise an error if the amplitude has no set function.
    pub fn evaluate(
        &self,
        pars: &ParMap,
        vars: &VarMap,
    ) -> Result<Complex64, Box<dyn Error + Send + Sync>> {
        if let Some(op) = &self.op {
            let op_lock = op.read();
            match &*op_lock {
                Operation::Add(a, b) => {
                    let res_a = a.evaluate(pars, vars)?;
                    let res_b = b.evaluate(pars, vars)?;
                    Ok(res_a + res_b)
                }
                Operation::Sub(a, b) => {
                    let res_a = a.evaluate(pars, vars)?;
                    let res_b = b.evaluate(pars, vars)?;
                    Ok(res_a - res_b)
                }
                Operation::Mul(a, b) => {
                    let res_a = a.evaluate(pars, vars)?;
                    let res_b = b.evaluate(pars, vars)?;
                    Ok(res_a * res_b)
                }
                Operation::Div(a, b) => {
                    let res_a = a.evaluate(pars, vars)?;
                    let res_b = b.evaluate(pars, vars)?;
                    Ok(res_a / res_b)
                }
                Operation::Neg(a) => Ok(-1.0 * a.evaluate(pars, vars)?),
                Operation::Sqrt(a) => Ok(a.evaluate(pars, vars)?.sqrt()),
                Operation::NormSquare(a) => Ok(a.evaluate(pars, vars)?.norm_sqr().into()),
                Operation::Real(a) => Ok(a.evaluate(pars, vars)?.re.into()),
                Operation::Imag(a) => Ok(a.evaluate(pars, vars)?.im.into()),
            }
        } else {
            self._evaluate(pars, vars)
        }
    }
    pub fn resolve_dependencies(&self, dataset: &mut Dataset) {
        if let Some(deps) = &self.dependencies {
            for dep in deps {
                dataset.resolve_dependencies(dep.clone(), false);
            }
        }
    }
    pub fn par_resolve_dependencies(&self, dataset: &mut Dataset) {
        if let Some(deps) = &self.dependencies {
            for dep in deps {
                dataset.par_resolve_dependencies(dep.clone(), false);
            }
        }
    }
    /// Note that `par_names` is, in general, not the same length as `par_values`. This is because
    /// it contains a single name for each complex variable while `par_values` will contain two
    /// `f64`s, one each for the real and imaginary part.
    ///
    /// # Panics
    ///
    /// Panics if a name in `par_names` is not found in the list of external parameter
    /// names for the amplitude.
    pub fn load_params(&self, par_vals: &[f64], par_names: &[&str]) {
        let mut i: usize = 0;
        let mut e_pars_lock = self.external_parameters.write();
        for e_name in par_names {
            if e_pars_lock.get_mut(*e_name).unwrap().value.is_scalar() {
                e_pars_lock.get_mut(*e_name).unwrap().value = ParameterValue::Scalar(par_vals[i]);
                i += 1;
            } else {
                e_pars_lock.get_mut(*e_name).unwrap().value =
                    ParameterValue::CScalar(Complex64::new(par_vals[i], par_vals[i + 1]));
                i += 2;
            }
        }
    }
    pub fn evaluate_on(&self, dataset: &Dataset) -> Vec<Complex64> {
        let parameter_map: ParMap = self.external_parameters.read().clone();
        dataset
            .entries
            .iter()
            .filter_map(|entry| self.evaluate(&parameter_map, entry).ok())
            .collect()
    }

    pub fn par_evaluate_on(&self, dataset: &Dataset) -> Vec<Complex64> {
        let parameter_map: ParMap = self.external_parameters.read().clone();
        dataset
            .entries
            .par_iter()
            .filter_map(|entry| self.evaluate(&parameter_map, entry).ok())
            .collect()
    }
    pub fn sqrt(&self) -> Amplitude<'a> {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Sqrt(self.clone())))),
            external_parameters: self.clone().external_parameters,
            dependencies: self.clone().dependencies,
            ..Default::default()
        }
    }
    pub fn norm_sqr(&self) -> Amplitude<'a> {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::NormSquare(self.clone())))),
            external_parameters: self.clone().external_parameters,
            dependencies: self.clone().dependencies,
            ..Default::default()
        }
    }
    pub fn re(&self) -> Amplitude<'a> {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Real(self.clone())))),
            external_parameters: self.clone().external_parameters,
            dependencies: self.clone().dependencies,
            ..Default::default()
        }
    }
    pub fn real(&self) -> Amplitude<'a> {
        self.re()
    }
    pub fn im(&self) -> Amplitude<'a> {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Imag(self.clone())))),
            external_parameters: self.clone().external_parameters,
            dependencies: self.clone().dependencies,
            ..Default::default()
        }
    }
    pub fn imag(&self) -> Amplitude<'a> {
        self.im()
    }
}

impl<'a> Add for Amplitude<'a> {
    type Output = Amplitude<'a>;

    fn add(self, rhs: Amplitude<'a>) -> Self::Output {
        let external_parameters = Arc::new(RwLock::new(
            self.external_parameters
                .read()
                .clone()
                .into_iter()
                .chain(rhs.external_parameters.read().clone())
                .collect(),
        ));
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };

        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Add(self, rhs)))),
            external_parameters,
            dependencies,
            ..Default::default()
        }
    }
}

impl<'a> Add for &'a Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn add(self, rhs: Self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Add(
                self.clone(),
                rhs.clone(),
            )))),
            external_parameters: Arc::new(RwLock::new(
                self.external_parameters
                    .read()
                    .clone()
                    .into_iter()
                    .chain(rhs.external_parameters.read().clone())
                    .collect(),
            )),
            dependencies: match (self.clone().dependencies, rhs.clone().dependencies) {
                (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
                (Some(deps), None) | (None, Some(deps)) => Some(deps),
                (None, None) => None,
            },
            ..Default::default()
        }
    }
}
impl<'a> Add<&'a Amplitude<'a>> for Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn add(self, rhs: &'a Self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Add(
                self.clone(),
                rhs.clone(),
            )))),
            external_parameters: Arc::new(RwLock::new(
                self.external_parameters
                    .read()
                    .clone()
                    .into_iter()
                    .chain(rhs.external_parameters.read().clone())
                    .collect(),
            )),
            dependencies: match (self.clone().dependencies, rhs.clone().dependencies) {
                (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
                (Some(deps), None) | (None, Some(deps)) => Some(deps),
                (None, None) => None,
            },
            ..Default::default()
        }
    }
}

impl<'a> Sub for Amplitude<'a> {
    type Output = Amplitude<'a>;

    fn sub(self, rhs: Amplitude<'a>) -> Self::Output {
        let external_parameters = Arc::new(RwLock::new(
            self.external_parameters
                .read()
                .clone()
                .into_iter()
                .chain(rhs.external_parameters.read().clone())
                .collect(),
        ));

        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };

        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Sub(self, rhs)))),
            external_parameters,
            dependencies,
            ..Default::default()
        }
    }
}

impl<'a> Sub for &'a Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn sub(self, rhs: Self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Sub(
                self.clone(),
                rhs.clone(),
            )))),

            external_parameters: Arc::new(RwLock::new(
                self.external_parameters
                    .read()
                    .clone()
                    .into_iter()
                    .chain(rhs.external_parameters.read().clone())
                    .collect(),
            )),

            dependencies: match (self.clone().dependencies, rhs.clone().dependencies) {
                (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
                (Some(deps), None) | (None, Some(deps)) => Some(deps),
                (None, None) => None,
            },
            ..Default::default()
        }
    }
}
impl<'a> Sub<&'a Amplitude<'a>> for Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn sub(self, rhs: &'a Self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Sub(
                self.clone(),
                rhs.clone(),
            )))),

            external_parameters: Arc::new(RwLock::new(
                self.external_parameters
                    .read()
                    .clone()
                    .into_iter()
                    .chain(rhs.external_parameters.read().clone())
                    .collect(),
            )),

            dependencies: match (self.clone().dependencies, rhs.clone().dependencies) {
                (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
                (Some(deps), None) | (None, Some(deps)) => Some(deps),
                (None, None) => None,
            },
            ..Default::default()
        }
    }
}

impl<'a> Mul for Amplitude<'a> {
    type Output = Amplitude<'a>;

    fn mul(self, rhs: Amplitude<'a>) -> Self::Output {
        let external_parameters = Arc::new(RwLock::new(
            self.external_parameters
                .read()
                .clone()
                .into_iter()
                .chain(rhs.external_parameters.read().clone())
                .collect(),
        ));
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };

        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Mul(self, rhs)))),
            external_parameters,
            dependencies,
            ..Default::default()
        }
    }
}

impl<'a> Mul for &'a Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn mul(self, rhs: Self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Mul(
                self.clone(),
                rhs.clone(),
            )))),
            external_parameters: Arc::new(RwLock::new(
                self.external_parameters
                    .read()
                    .clone()
                    .into_iter()
                    .chain(rhs.external_parameters.read().clone())
                    .collect(),
            )),

            dependencies: match (self.clone().dependencies, rhs.clone().dependencies) {
                (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
                (Some(deps), None) | (None, Some(deps)) => Some(deps),
                (None, None) => None,
            },
            ..Default::default()
        }
    }
}
impl<'a> Mul<&'a Amplitude<'a>> for Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn mul(self, rhs: &'a Self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Mul(
                self.clone(),
                rhs.clone(),
            )))),
            external_parameters: Arc::new(RwLock::new(
                self.external_parameters
                    .read()
                    .clone()
                    .into_iter()
                    .chain(rhs.external_parameters.read().clone())
                    .collect(),
            )),

            dependencies: match (self.clone().dependencies, rhs.clone().dependencies) {
                (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
                (Some(deps), None) | (None, Some(deps)) => Some(deps),
                (None, None) => None,
            },
            ..Default::default()
        }
    }
}
impl<'a> Div for Amplitude<'a> {
    type Output = Amplitude<'a>;

    fn div(self, rhs: Amplitude<'a>) -> Self::Output {
        let external_parameters = Arc::new(RwLock::new(
            self.external_parameters
                .read()
                .clone()
                .into_iter()
                .chain(rhs.external_parameters.read().clone())
                .collect(),
        ));
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };

        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Div(self, rhs)))),
            external_parameters,
            dependencies,
            ..Default::default()
        }
    }
}
impl<'a> Div for &'a Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn div(self, rhs: Self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Div(
                self.clone(),
                rhs.clone(),
            )))),
            external_parameters: Arc::new(RwLock::new(
                self.external_parameters
                    .read()
                    .clone()
                    .into_iter()
                    .chain(rhs.external_parameters.read().clone())
                    .collect(),
            )),

            dependencies: match (self.clone().dependencies, rhs.clone().dependencies) {
                (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
                (Some(deps), None) | (None, Some(deps)) => Some(deps),
                (None, None) => None,
            },
            ..Default::default()
        }
    }
}
impl<'a> Div<&'a Amplitude<'a>> for Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn div(self, rhs: &'a Self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Div(
                self.clone(),
                rhs.clone(),
            )))),
            external_parameters: Arc::new(RwLock::new(
                self.external_parameters
                    .read()
                    .clone()
                    .into_iter()
                    .chain(rhs.external_parameters.read().clone())
                    .collect(),
            )),

            dependencies: match (self.clone().dependencies, rhs.clone().dependencies) {
                (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
                (Some(deps), None) | (None, Some(deps)) => Some(deps),
                (None, None) => None,
            },
            ..Default::default()
        }
    }
}

impl<'a> Neg for &'a Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn neg(self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Neg(self.clone())))),
            external_parameters: self.clone().external_parameters,
            dependencies: self.clone().dependencies,
            ..Default::default()
        }
    }
}
impl<'a> Neg for Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn neg(self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Neg(self.clone())))),
            external_parameters: self.clone().external_parameters,
            dependencies: self.clone().dependencies,
            ..Default::default()
        }
    }
}

impl<'a> From<f64> for Amplitude<'a> {
    fn from(value: f64) -> Self {
        Amplitude::new(&value.to_string(), move |_, _| Ok(value.into()), None, None)
    }
}

impl<'a> From<Complex64> for Amplitude<'a> {
    fn from(value: Complex64) -> Self {
        Amplitude::new(&value.to_string(), move |_, _| Ok(value), None, None)
    }
}

pub trait AmplitudeBuilder<'a> {
    fn into_amplitude(self) -> Amplitude<'a>;
    fn assign(self, parameters: &[Parameter<'a>]) -> Amplitude<'a>
    where
        Self: Sized,
    {
        let amplitude = self.into_amplitude();
        let internal_names = amplitude.internal_parameters.read().clone();
        for (e_par, i_name) in parameters.iter().zip(internal_names.iter()) {
            amplitude.assign(e_par, i_name);
        }
        amplitude
    }
}

#[derive(Variantly, Clone, Copy, Debug)]
pub enum ParameterValue {
    Scalar(f64),
    #[variantly(rename = "cscalar")]
    CScalar(Complex64),
}

#[derive(Clone, Copy, Debug)]
pub struct Parameter<'a> {
    pub name: &'a str,
    pub value: ParameterValue,
}

impl<'a> Hash for Parameter<'a> {
    /// This ensures the hash lookup only depends on the name of the parameter, not its value
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl<'a> PartialEq for Parameter<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl<'a> Eq for Parameter<'a> {}

impl<'a> Parameter<'a> {
    pub fn new(name: &str, value: ParameterValue) -> Parameter {
        Parameter { name, value }
    }
}

impl<'a> From<Parameter<'a>> for Amplitude<'a> {
    fn from(par: Parameter<'a>) -> Self {
        Amplitude::new(
            par.name,
            |pars: &ParMap, _vars: &VarMap| {
                Ok(match pars["parameter"].value {
                    ParameterValue::Scalar(val) => Complex64::from(val),
                    ParameterValue::CScalar(val) => val,
                })
            },
            Some(vec!["parameter"]),
            None,
        )
    }
}

// #[derive(Clone, Copy, Variantly)]
// pub enum ParameterType<'a> {
//     Scalar(Parameter<'a>),
//     #[variantly(rename = "cscalar")]
//     CScalar(ComplexParameter<'a>),
// }
//
// impl<'a> From<Parameter<'a>> for ParameterType<'a> {
//     fn from(par: Parameter<'a>) -> Self {
//         ParameterType::Scalar(par)
//     }
// }
//
// impl<'a> From<ComplexParameter<'a>> for ParameterType<'a> {
//     fn from(par: ComplexParameter<'a>) -> Self {
//         ParameterType::CScalar(par)
//     }
// }
//
// #[derive(Clone, Copy, new)]
// pub struct Parameter<'a> {
//     name: &'a str,
//     pub value: f64,
//     #[new(default)]
//     lower_bound: Option<f64>,
//     #[new(default)]
//     upper_bound: Option<f64>,
//     #[new(value = "false")]
//     fixed: bool,
// }
//
// impl<'a> From<Parameter<'a>> for f64 {
//     fn from(par: Parameter) -> Self {
//         par.value()
//     }
// }
// impl<'a> From<Parameter<'a>> for Complex64 {
//     fn from(par: Parameter) -> Self {
//         par.value().into()
//     }
// }
//
// impl<'a> Parameter<'a> {
//     pub fn value(&self) -> f64 {
//         self.value
//     }
//
//     pub fn as_complex(&self) -> Complex64 {
//         self.value.into()
//     }
//
//     pub fn with_bounds(
//         mut self,
//         lower_bound: Option<f64>,
//         upper_bound: Option<f64>,
//     ) -> Parameter<'a> {
//         self.lower_bound = lower_bound;
//         self.upper_bound = upper_bound;
//         self
//     }
//     pub fn fix(mut self) -> Parameter<'a> {
//         self.fixed = true;
//         self
//     }
//     pub fn free(mut self) -> Parameter<'a> {
//         self.fixed = false;
//         self
//     }
//     pub fn as_amp(&self) -> Amplitude {
//         Amplitude::new(
//             self.name,
//             |pars: &ParMap, _vars: &VarMap| Ok(pars["parameter"].scalar().unwrap().as_complex()),
//             None,
//         )
//         .with_pars(vec!["parameter".to_string()])
//         .link(self.name, "parameter")
//     }
// }
//
// #[derive(Clone, Copy)]
// pub enum Coordinates {
//     Cartesian,
//     Polar,
// }
//
// #[derive(Clone, Copy)]
// pub struct ComplexParameter<'a> {
//     pub name: &'a str,
//     pub a: f64,
//     pub b: f64,
//     pub coordinates: Coordinates,
// }
//
// impl<'a> From<ComplexParameter<'a>> for Complex64 {
//     fn from(par: ComplexParameter) -> Self {
//         par.value()
//     }
// }
//
// impl<'a> ComplexParameter<'a> {
//     pub fn value(&self) -> Complex64 {
//         match self.coordinates {
//             Coordinates::Cartesian => Complex64::new(self.a, self.b),
//             Coordinates::Polar => Complex64::from_polar(self.a, self.b),
//         }
//     }
//
//     pub fn as_amp(&self) -> Amplitude {
//         Amplitude::new(
//             self.name,
//             |pars: &ParMap, _vars: &VarMap| Ok(pars["parameter"].cscalar().unwrap().value()),
//             None,
//         )
//         .with_pars(vec!["parameter".to_string()])
//         .link(self.name, "parameter")
//     }
// }

#[derive(Clone)]
pub struct Variable {
    pub name: Arc<String>,
    pub function: ArcVarFn,
    pub dependencies: Option<Vec<Variable>>,
}

impl std::fmt::Debug for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VAR({:?} -> {:?})", self.name, self.dependencies)
    }
}

impl Variable {
    pub fn new<F>(name: &str, function: F, dependencies: Option<Vec<Variable>>) -> Self
    where
        F: 'static + Fn(&VarMap) -> FieldType + Sync + Send,
    {
        Variable {
            name: Arc::new(name.to_string()),
            function: Arc::new(RwLock::new(function)),
            dependencies,
        }
    }
}

pub trait VariableBuilder {
    fn into_variable(self) -> Variable;
}
