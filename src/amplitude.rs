use anyinput::anyinput;
use dashmap::DashMap;
use derive_builder::Builder;
use num_traits::pow::Pow;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::{error::Error, sync::Arc};
use variantly::Variantly;

use num_complex::Complex64;

use crate::dataset::DatasetError;
use crate::prelude::{Dataset, Entry, Resolve};
use crate::variable::Variable;

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

#[derive(Clone)]
enum Operation<'a> {
    Add(Amplitude<'a>, Amplitude<'a>),
    Sub(Amplitude<'a>, Amplitude<'a>),
    Mul(Amplitude<'a>, Amplitude<'a>),
    Div(Amplitude<'a>, Amplitude<'a>),
    Pow(Amplitude<'a>, Amplitude<'a>),
    Neg(Amplitude<'a>),
    Sqrt(Amplitude<'a>),
    NormSquare(Amplitude<'a>),
    Real(Amplitude<'a>),
    Imag(Amplitude<'a>),
}

pub type ParMap<'a> = DashMap<String, Parameter<'a>>;
pub type SendableAmpFn =
    dyn Fn(&ParMap, &Entry) -> Result<Complex64, Box<dyn Error + Send + Sync>> + Send + Sync;
pub type ArcAmpFn = Arc<RwLock<SendableAmpFn>>;

#[derive(Default, Clone, Builder)]
pub struct Amplitude<'a> {
    #[builder(setter(custom))]
    pub name: Arc<String>,
    #[builder(setter(custom))]
    function: Option<ArcAmpFn>,
    #[builder(setter(custom), default = "Arc::new(RwLock::new(Vec::default()))")]
    internal_parameters: Arc<RwLock<Vec<String>>>,
    #[builder(setter(custom), default)]
    pub dependencies: Option<Vec<Variable>>,
    #[builder(setter(skip))]
    pub external_parameters: Arc<DashMap<String, Parameter<'a>>>,
    #[builder(setter(skip))]
    parameter_mappings: Arc<DashMap<Parameter<'a>, String>>,
    #[builder(setter(skip))]
    op: Option<Arc<RwLock<Operation<'a>>>>,
}

impl<'a> AmplitudeBuilder<'a> {
    #[anyinput]
    pub fn name(&mut self, value: AnyString) -> &mut Self {
        self.name = Some(Arc::new(value.to_string()));
        self
    }

    pub fn function<F>(&mut self, f: F) -> &mut Self
    where
        F: 'static
            + Fn(&ParMap, &Entry) -> Result<Complex64, Box<dyn Error + Send + Sync>>
            + Sync
            + Send,
    {
        self.function = Some(Some(Arc::new(RwLock::new(f))));
        self
    }
    #[anyinput]
    pub fn internal_parameters(&mut self, parameter_names: AnyIter<AnyString>) -> &mut Self {
        self.internal_parameters = Some(Arc::new(RwLock::new(
            parameter_names
                .into_iter()
                .map(|s| s.as_ref().to_string())
                .collect::<Vec<String>>(),
        )));
        self
    }
    #[anyinput]
    pub fn dependencies(&mut self, variables: AnyIter<Variable>) -> &mut Self {
        self.dependencies = Some(Some(variables.into_iter().collect::<Vec<_>>()));
        self
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
                .insert(*external_par, internal_name.to_string());
            self.external_parameters
                .insert(external_par.name.to_string(), *external_par);
        } else {
            panic!("Name not found!");
        }
    }

    fn _evaluate(
        &self,
        pars: &ParMap,
        vars: &Entry,
    ) -> Result<Complex64, Box<dyn Error + Send + Sync>> {
        let internal_pars = self
            .parameter_mappings
            .iter()
            .filter_map(|entry| {
                pars.get(entry.key().name)
                    .map(|par| (entry.value().clone(), *par))
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
        vars: &Entry,
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
                Operation::Pow(a, b) => {
                    let res_a = a.evaluate(pars, vars)?;
                    let res_b = b.evaluate(pars, vars)?;
                    Ok(res_a.powc(res_b))
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

    pub fn load_params(&self, par_vals: &[f64], par_names: &[&str]) {
        //! Note that `par_names` is, in general, not the same length as `par_values`. This is because
        //! it contains a single name for each complex variable while `par_values` will contain two
        //! `f64`s, one each for the real and imaginary part.
        //!
        //! # Panics
        //!
        //! Panics if a name in `par_names` is not found in the list of external parameter
        //! names for the amplitude.
        let mut i: usize = 0;
        for e_name in par_names {
            if self
                .external_parameters
                .get_mut(*e_name)
                .unwrap()
                .value
                .is_scalar()
            {
                self.external_parameters.get_mut(*e_name).unwrap().value =
                    ParameterValue::Scalar(par_vals[i]);
                i += 1;
            } else {
                self.external_parameters.get_mut(*e_name).unwrap().value =
                    ParameterValue::CScalar(Complex64::new(par_vals[i], par_vals[i + 1]));
                i += 2;
            }
        }
    }

    pub fn resolve(&self, dataset: &mut Dataset) -> Result<(), DatasetError> {
        if let Some(deps) = &self.dependencies {
            for dep in deps {
                match dep {
                    Variable::Scalar(var) => var.resolve(dataset, false)?,
                    Variable::CScalar(var) => var.resolve(dataset, false)?,
                    Variable::Vector(var) => var.resolve(dataset, false)?,
                    Variable::CVector(var) => var.resolve(dataset, false)?,
                    Variable::Matrix(var) => var.resolve(dataset, false)?,
                    Variable::CMatrix(var) => var.resolve(dataset, false)?,
                    Variable::Momentum(var) => var.resolve(dataset, false)?,
                    Variable::Momenta(var) => var.resolve(dataset, false)?,
                };
            }
        }
        Ok(())
    }
    pub fn resolve_par(&self, dataset: &mut Dataset) -> Result<(), DatasetError> {
        if let Some(deps) = &self.dependencies {
            for dep in deps {
                match dep {
                    Variable::Scalar(var) => var.resolve_par(dataset, false)?,
                    Variable::CScalar(var) => var.resolve_par(dataset, false)?,
                    Variable::Vector(var) => var.resolve_par(dataset, false)?,
                    Variable::CVector(var) => var.resolve_par(dataset, false)?,
                    Variable::Matrix(var) => var.resolve_par(dataset, false)?,
                    Variable::CMatrix(var) => var.resolve_par(dataset, false)?,
                    Variable::Momentum(var) => var.resolve_par(dataset, false)?,
                    Variable::Momenta(var) => var.resolve_par(dataset, false)?,
                };
            }
        }
        Ok(())
    }
    pub fn evaluate_on(&self, dataset: &Dataset) -> Vec<Complex64> {
        dataset
            .entries
            .iter()
            .filter_map(|entry| self.evaluate(&self.external_parameters, entry).ok())
            .collect()
    }

    pub fn evaluate_on_par(&self, dataset: &Dataset) -> Vec<Complex64> {
        dataset
            .entries
            .par_iter()
            .filter_map(|entry| self.evaluate(&self.external_parameters, entry).ok())
            .collect()
    }
    pub fn sqrt(&self) -> Amplitude<'a> {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Sqrt(self.clone())))),
            external_parameters: self.external_parameters.clone(),
            dependencies: self.dependencies.clone(),
            ..Default::default()
        }
    }
    pub fn norm_sqr(&self) -> Amplitude<'a> {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::NormSquare(self.clone())))),
            external_parameters: self.external_parameters.clone(),
            dependencies: self.dependencies.clone(),
            ..Default::default()
        }
    }
    pub fn re(&self) -> Amplitude<'a> {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Real(self.clone())))),
            external_parameters: self.external_parameters.clone(),
            dependencies: self.dependencies.clone(),
            ..Default::default()
        }
    }
    pub fn real(&self) -> Amplitude<'a> {
        self.re()
    }
    pub fn im(&self) -> Amplitude<'a> {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Imag(self.clone())))),
            external_parameters: self.external_parameters.clone(),
            dependencies: self.dependencies.clone(),
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
        let external_parameters = DashMap::new();
        self.external_parameters
            .iter()
            .chain(rhs.external_parameters.iter())
            .for_each(|entry| {
                external_parameters.insert(entry.key().to_string(), *entry.value());
            });
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Add(self, rhs)))),
            external_parameters: Arc::new(external_parameters),
            dependencies,
            ..Default::default()
        }
    }
}
impl<'a> Add for &'a Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn add(self, rhs: Self) -> Self::Output {
        let external_parameters = DashMap::new();
        self.external_parameters
            .iter()
            .chain(rhs.external_parameters.iter())
            .for_each(|entry| {
                external_parameters.insert(entry.key().to_string(), *entry.value());
            });
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Add(
                self.clone(),
                rhs.clone(),
            )))),
            external_parameters: Arc::new(external_parameters),
            dependencies,
            ..Default::default()
        }
    }
}
impl<'a> Add<&'a Amplitude<'a>> for Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn add(self, rhs: &'a Self) -> Self::Output {
        let external_parameters = DashMap::new();
        self.external_parameters
            .iter()
            .chain(rhs.external_parameters.iter())
            .for_each(|entry| {
                external_parameters.insert(entry.key().to_string(), *entry.value());
            });
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Add(self, rhs.clone())))),
            external_parameters: Arc::new(external_parameters),
            dependencies,
            ..Default::default()
        }
    }
}

impl<'a> Sub for Amplitude<'a> {
    type Output = Amplitude<'a>;

    fn sub(self, rhs: Amplitude<'a>) -> Self::Output {
        let external_parameters = DashMap::new();
        self.external_parameters
            .iter()
            .chain(rhs.external_parameters.iter())
            .for_each(|entry| {
                external_parameters.insert(entry.key().to_string(), *entry.value());
            });
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Sub(self, rhs)))),
            external_parameters: Arc::new(external_parameters),
            dependencies,
            ..Default::default()
        }
    }
}
impl<'a> Sub for &'a Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn sub(self, rhs: Self) -> Self::Output {
        let external_parameters = DashMap::new();
        self.external_parameters
            .iter()
            .chain(rhs.external_parameters.iter())
            .for_each(|entry| {
                external_parameters.insert(entry.key().to_string(), *entry.value());
            });
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Sub(
                self.clone(),
                rhs.clone(),
            )))),
            external_parameters: Arc::new(external_parameters),
            dependencies,
            ..Default::default()
        }
    }
}
impl<'a> Sub<&'a Amplitude<'a>> for Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn sub(self, rhs: &'a Self) -> Self::Output {
        let external_parameters = DashMap::new();
        self.external_parameters
            .iter()
            .chain(rhs.external_parameters.iter())
            .for_each(|entry| {
                external_parameters.insert(entry.key().to_string(), *entry.value());
            });
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Sub(self, rhs.clone())))),
            external_parameters: Arc::new(external_parameters),
            dependencies,
            ..Default::default()
        }
    }
}

impl<'a> Mul for Amplitude<'a> {
    type Output = Amplitude<'a>;

    fn mul(self, rhs: Amplitude<'a>) -> Self::Output {
        let external_parameters = DashMap::new();
        self.external_parameters
            .iter()
            .chain(rhs.external_parameters.iter())
            .for_each(|entry| {
                external_parameters.insert(entry.key().to_string(), *entry.value());
            });
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Mul(self, rhs)))),
            external_parameters: Arc::new(external_parameters),
            dependencies,
            ..Default::default()
        }
    }
}
impl<'a> Mul for &'a Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn mul(self, rhs: Self) -> Self::Output {
        let external_parameters = DashMap::new();
        self.external_parameters
            .iter()
            .chain(rhs.external_parameters.iter())
            .for_each(|entry| {
                external_parameters.insert(entry.key().to_string(), *entry.value());
            });
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Mul(
                self.clone(),
                rhs.clone(),
            )))),
            external_parameters: Arc::new(external_parameters),
            dependencies,
            ..Default::default()
        }
    }
}
impl<'a> Mul<&'a Amplitude<'a>> for Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn mul(self, rhs: &'a Self) -> Self::Output {
        let external_parameters = DashMap::new();
        self.external_parameters
            .iter()
            .chain(rhs.external_parameters.iter())
            .for_each(|entry| {
                external_parameters.insert(entry.key().to_string(), *entry.value());
            });
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Mul(self, rhs.clone())))),
            external_parameters: Arc::new(external_parameters),
            dependencies,
            ..Default::default()
        }
    }
}

impl<'a> Div for Amplitude<'a> {
    type Output = Amplitude<'a>;

    fn div(self, rhs: Amplitude<'a>) -> Self::Output {
        let external_parameters = DashMap::new();
        self.external_parameters
            .iter()
            .chain(rhs.external_parameters.iter())
            .for_each(|entry| {
                external_parameters.insert(entry.key().to_string(), *entry.value());
            });
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Div(self, rhs)))),
            external_parameters: Arc::new(external_parameters),
            dependencies,
            ..Default::default()
        }
    }
}
impl<'a> Div for &'a Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn div(self, rhs: Self) -> Self::Output {
        let external_parameters = DashMap::new();
        self.external_parameters
            .iter()
            .chain(rhs.external_parameters.iter())
            .for_each(|entry| {
                external_parameters.insert(entry.key().to_string(), *entry.value());
            });
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Div(
                self.clone(),
                rhs.clone(),
            )))),
            external_parameters: Arc::new(external_parameters),
            dependencies,
            ..Default::default()
        }
    }
}
impl<'a> Div<&'a Amplitude<'a>> for Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn div(self, rhs: &'a Self) -> Self::Output {
        let external_parameters = DashMap::new();
        self.external_parameters
            .iter()
            .chain(rhs.external_parameters.iter())
            .for_each(|entry| {
                external_parameters.insert(entry.key().to_string(), *entry.value());
            });
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Div(self, rhs.clone())))),
            external_parameters: Arc::new(external_parameters),
            dependencies,
            ..Default::default()
        }
    }
}

impl<'a> Pow<Amplitude<'a>> for Amplitude<'a> {
    type Output = Amplitude<'a>;

    fn pow(self, rhs: Amplitude<'a>) -> Self::Output {
        let external_parameters = DashMap::new();
        self.external_parameters
            .iter()
            .chain(rhs.external_parameters.iter())
            .for_each(|entry| {
                external_parameters.insert(entry.key().to_string(), *entry.value());
            });
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Pow(self, rhs)))),
            external_parameters: Arc::new(external_parameters),
            dependencies,
            ..Default::default()
        }
    }
}
impl<'a> Pow<Self> for &'a Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn pow(self, rhs: Self) -> Self::Output {
        let external_parameters = DashMap::new();
        self.external_parameters
            .iter()
            .chain(rhs.external_parameters.iter())
            .for_each(|entry| {
                external_parameters.insert(entry.key().to_string(), *entry.value());
            });
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Pow(
                self.clone(),
                rhs.clone(),
            )))),
            external_parameters: Arc::new(external_parameters),
            dependencies,
            ..Default::default()
        }
    }
}
impl<'a> Pow<&'a Amplitude<'a>> for Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn pow(self, rhs: &'a Self) -> Self::Output {
        let external_parameters = DashMap::new();
        self.external_parameters
            .iter()
            .chain(rhs.external_parameters.iter())
            .for_each(|entry| {
                external_parameters.insert(entry.key().to_string(), *entry.value());
            });
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Pow(self, rhs.clone())))),
            external_parameters: Arc::new(external_parameters),
            dependencies,
            ..Default::default()
        }
    }
}

impl<'a> Neg for &'a Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn neg(self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Neg(self.clone())))),
            external_parameters: self.external_parameters.clone(),
            dependencies: self.dependencies.clone(),
            ..Default::default()
        }
    }
}
impl<'a> Neg for Amplitude<'a> {
    type Output = Amplitude<'a>;
    fn neg(self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Neg(self.clone())))),
            external_parameters: self.external_parameters,
            dependencies: self.dependencies,
            ..Default::default()
        }
    }
}

impl<'a> From<f64> for Amplitude<'a> {
    fn from(value: f64) -> Self {
        AmplitudeBuilder::default()
            .name(value.to_string())
            .function(move |_, _| Ok(value.into()))
            .build()
            .unwrap()
    }
}
impl<'a> From<Complex64> for Amplitude<'a> {
    fn from(value: Complex64) -> Self {
        AmplitudeBuilder::default()
            .name(value.to_string())
            .function(move |_, _| Ok(value))
            .build()
            .unwrap()
    }
}

pub trait IntoAmplitude<'a> {
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
        AmplitudeBuilder::default()
            .name(par.name)
            .function(|pars: &ParMap, _vars: &Entry| {
                Ok(match pars.get("parameter").unwrap().value {
                    ParameterValue::Scalar(val) => Complex64::from(val),
                    ParameterValue::CScalar(val) => val,
                })
            })
            .internal_parameters(["parameter"])
            .build()
            .unwrap()
    }
}
