use rayon::prelude::*;
use std::collections::HashSet;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::RwLock;
use std::{collections::HashMap, error::Error, sync::Arc};

use num_complex::Complex64;

use crate::prelude::{Dataset, FieldType};

#[macro_export]
macro_rules! par {
    ($name:expr, $value:expr) => {
        ParameterType::Scalar(Parameter::new($name.to_string(), $value))
    };
}

#[macro_export]
macro_rules! cpar {
    ($name:expr, $value_re:expr, $value_im:expr) => {
        ParameterType::CScalar(ComplexParameter {
            name: $name.to_string(),
            a: Parameter::new(format!("{} (re)", $name), $value_re),
            b: Parameter::new(format!("{} (im)", $name), $value_im),
            coordinates: Coordinates::Cartesian,
        })
    };
    ($name:expr, $value_re:expr, $value_im:expr, Polar) => {
        ParameterType::CScalar(ComplexParameter {
            name: $name.to_string(),
            a: Parameter::new(format!("{} (r)", $name), $value_re),
            b: Parameter::new(format!("{} (theta)", $name), $value_im),
            coordinates: Coordinates::Polar,
        })
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
enum Operation {
    Add(Amplitude, Amplitude),
    Sub(Amplitude, Amplitude),
    Mul(Amplitude, Amplitude),
    Div(Amplitude, Amplitude),
    Neg(Amplitude),
    Sqrt(Amplitude),
    NormSquare(Amplitude),
    Real(Amplitude),
    Imag(Amplitude),
}

pub type ParMap = HashMap<String, ParameterType>;
pub type VarMap = HashMap<String, FieldType>;
pub type SendableAmpFn =
    dyn Fn(&ParMap, &VarMap) -> Result<Complex64, Box<dyn Error + Send + Sync>> + Send + Sync;
pub type ArcAmpFn = Arc<RwLock<SendableAmpFn>>;
pub type SendableVarFn = dyn Fn(&VarMap) -> FieldType + Send + Sync;
pub type ArcVarFn = Arc<RwLock<SendableVarFn>>;

#[derive(Default, Clone)]
pub struct Amplitude {
    name: Arc<String>,
    function: Option<ArcAmpFn>,
    parameters: Option<Arc<RwLock<HashSet<String>>>>,
    parameter_mappings: Arc<RwLock<HashMap<String, String>>>,
    op: Option<Arc<RwLock<Operation>>>,
    dependencies: Option<Vec<Variable>>,
}

impl std::fmt::Debug for Amplitude {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} ->\n\tpars = {:?}\n\tmap = {:?}\n\tdeps = {:?}",
            self.name, self.parameters, self.parameter_mappings, self.dependencies
        )
    }
}

impl Amplitude {
    pub fn new<F>(name: String, function: F) -> Self
    where
        F: 'static
            + Fn(&ParMap, &VarMap) -> Result<Complex64, Box<dyn Error + Send + Sync>>
            + Sync
            + Send,
    {
        Amplitude {
            name: Arc::new(name),
            function: Some(Arc::new(RwLock::new(function))),
            ..Default::default()
        }
    }

    fn with_pars(mut self, names: Vec<String>) -> Self {
        self.parameters = Some(Arc::new(RwLock::new(names.into_iter().collect())));
        self
    }

    pub fn with_deps(mut self, dependencies: Vec<Variable>) -> Self {
        self.dependencies = Some(dependencies);
        self
    }

    pub fn map(self, external_name: String, internal_name: String) -> Self {
        if let Some(pars_arc) = &self.parameters {
            let pars = pars_arc.read().expect("Failed to read pars");
            println!("Names:");
            println!("{internal_name}");
            println!("{external_name}");
            println!("{:?}", pars);
            if pars.contains(&internal_name) {
                let mut mappings_lock = self
                    .parameter_mappings
                    .write()
                    .expect("Failed to access parameter_mappings");
                mappings_lock.insert(external_name, internal_name);
            } else {
                panic!("Name not found!");
            }
        } else {
            panic!("No params!");
        }
        self
    }
    pub fn link(self, external_name: String, internal_name: String) -> Self {
        self.map(external_name, internal_name)
    }

    fn _evaluate(
        &self,
        pars: &ParMap,
        vars: &VarMap,
    ) -> Result<Complex64, Box<dyn Error + Send + Sync>> {
        let internal_pars = self
            .parameter_mappings
            .read()
            .expect("Failed to acquire lock on mappings")
            .iter()
            .filter_map(|(external, internal)| {
                pars.get(external)
                    .map(|par| (internal.clone(), par.clone()))
            })
            .collect();
        if let Some(ref func_arc) = self.function {
            let func_rwlock = func_arc.read().expect("Failed to read the RwLock");
            func_rwlock(&internal_pars, vars)
        } else {
            Err("Function is not set".into())
        }
    }

    /// TODO: need to finish writing the resolver and then implement the typical Scalar/CScalar
    /// amps, along with something that turns pars into amps maybe, I think we can do that now.
    /// Finally, need to make a Var -> Amp system and then wrap it all together.

    pub fn evaluate(
        &self,
        pars: &ParMap,
        vars: &VarMap,
    ) -> Result<Complex64, Box<dyn Error + Send + Sync>> {
        if let Some(op) = &self.op {
            let op_lock = op.read().expect("Failed to read op");
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
    pub fn evaluate_on(&self, pars: &[&ParameterType], dataset: &Dataset) -> Vec<Complex64> {
        let parameter_map: ParMap = pars
            .iter()
            .map(|&param| {
                (
                    match &param {
                        ParameterType::Scalar(par) => par.name.clone(),
                        ParameterType::CScalar(par) => par.name.clone(),
                    },
                    param.clone(),
                )
            })
            .collect();
        let mut output: Vec<Complex64> = Vec::new();
        for entry in &dataset.entries {
            let res = self.evaluate(&parameter_map, entry);
            if let Ok(res_val) = res {
                output.push(res_val);
            } else {
                println!("{res:?}")
            }
        }
        output
    }

    pub fn par_evaluate_on(&self, pars: &[&ParameterType], dataset: &Dataset) -> Vec<Complex64> {
        let parameter_map: ParMap = pars
            .iter()
            .map(|&param| {
                (
                    match &param {
                        ParameterType::Scalar(par) => par.name.clone(),
                        ParameterType::CScalar(par) => par.name.clone(),
                    },
                    param.clone(),
                )
            })
            .collect();
        dataset
            .entries
            .par_iter()
            .filter_map(|entry| self.evaluate(&parameter_map, entry).ok())
            .collect()
    }
    pub fn sqrt(&self) -> Amplitude {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Sqrt(self.clone())))),
            dependencies: self.clone().dependencies,
            ..Default::default()
        }
    }
    pub fn norm_sqr(&self) -> Amplitude {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::NormSquare(self.clone())))),
            dependencies: self.clone().dependencies,
            ..Default::default()
        }
    }
    pub fn re(&self) -> Amplitude {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Real(self.clone())))),
            dependencies: self.clone().dependencies,
            ..Default::default()
        }
    }
    pub fn real(&self) -> Amplitude {
        self.re()
    }
    pub fn im(&self) -> Amplitude {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Imag(self.clone())))),
            dependencies: self.clone().dependencies,
            ..Default::default()
        }
    }
    pub fn imag(&self) -> Amplitude {
        self.im()
    }
}

impl Add for Amplitude {
    type Output = Amplitude;

    fn add(self, rhs: Amplitude) -> Self::Output {
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };

        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Add(self, rhs)))),
            dependencies,
            ..Default::default()
        }
    }
}

impl<'a> Add for &'a Amplitude {
    type Output = Amplitude;
    fn add(self, rhs: Self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Add(
                self.clone(),
                rhs.clone(),
            )))),
            dependencies: match (self.clone().dependencies, rhs.clone().dependencies) {
                (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
                (Some(deps), None) | (None, Some(deps)) => Some(deps),
                (None, None) => None,
            },
            ..Default::default()
        }
    }
}
impl<'a> Add<&'a Amplitude> for Amplitude {
    type Output = Amplitude;
    fn add(self, rhs: &'a Self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Add(
                self.clone(),
                rhs.clone(),
            )))),
            dependencies: match (self.clone().dependencies, rhs.clone().dependencies) {
                (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
                (Some(deps), None) | (None, Some(deps)) => Some(deps),
                (None, None) => None,
            },
            ..Default::default()
        }
    }
}

impl Sub for Amplitude {
    type Output = Amplitude;

    fn sub(self, rhs: Amplitude) -> Self::Output {
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };

        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Sub(self, rhs)))),
            dependencies,
            ..Default::default()
        }
    }
}

impl<'a> Sub for &'a Amplitude {
    type Output = Amplitude;
    fn sub(self, rhs: Self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Sub(
                self.clone(),
                rhs.clone(),
            )))),
            dependencies: match (self.clone().dependencies, rhs.clone().dependencies) {
                (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
                (Some(deps), None) | (None, Some(deps)) => Some(deps),
                (None, None) => None,
            },
            ..Default::default()
        }
    }
}
impl<'a> Sub<&'a Amplitude> for Amplitude {
    type Output = Amplitude;
    fn sub(self, rhs: &'a Self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Sub(
                self.clone(),
                rhs.clone(),
            )))),
            dependencies: match (self.clone().dependencies, rhs.clone().dependencies) {
                (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
                (Some(deps), None) | (None, Some(deps)) => Some(deps),
                (None, None) => None,
            },
            ..Default::default()
        }
    }
}

impl Mul for Amplitude {
    type Output = Amplitude;

    fn mul(self, rhs: Amplitude) -> Self::Output {
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };

        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Mul(self, rhs)))),
            dependencies,
            ..Default::default()
        }
    }
}

impl<'a> Mul for &'a Amplitude {
    type Output = Amplitude;
    fn mul(self, rhs: Self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Mul(
                self.clone(),
                rhs.clone(),
            )))),
            dependencies: match (self.clone().dependencies, rhs.clone().dependencies) {
                (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
                (Some(deps), None) | (None, Some(deps)) => Some(deps),
                (None, None) => None,
            },
            ..Default::default()
        }
    }
}
impl<'a> Mul<&'a Amplitude> for Amplitude {
    type Output = Amplitude;
    fn mul(self, rhs: &'a Self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Mul(
                self.clone(),
                rhs.clone(),
            )))),
            dependencies: match (self.clone().dependencies, rhs.clone().dependencies) {
                (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
                (Some(deps), None) | (None, Some(deps)) => Some(deps),
                (None, None) => None,
            },
            ..Default::default()
        }
    }
}
impl Div for Amplitude {
    type Output = Amplitude;

    fn div(self, rhs: Amplitude) -> Self::Output {
        let dependencies = match (self.dependencies.clone(), rhs.dependencies.clone()) {
            (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
            (Some(deps), None) | (None, Some(deps)) => Some(deps),
            (None, None) => None,
        };

        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Div(self, rhs)))),
            dependencies,
            ..Default::default()
        }
    }
}
impl<'a> Div for &'a Amplitude {
    type Output = Amplitude;
    fn div(self, rhs: Self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Div(
                self.clone(),
                rhs.clone(),
            )))),
            dependencies: match (self.clone().dependencies, rhs.clone().dependencies) {
                (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
                (Some(deps), None) | (None, Some(deps)) => Some(deps),
                (None, None) => None,
            },
            ..Default::default()
        }
    }
}
impl<'a> Div<&'a Amplitude> for Amplitude {
    type Output = Amplitude;
    fn div(self, rhs: &'a Self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Div(
                self.clone(),
                rhs.clone(),
            )))),
            dependencies: match (self.clone().dependencies, rhs.clone().dependencies) {
                (Some(deps_a), Some(deps_b)) => Some(deps_a.into_iter().chain(deps_b).collect()),
                (Some(deps), None) | (None, Some(deps)) => Some(deps),
                (None, None) => None,
            },
            ..Default::default()
        }
    }
}

impl<'a> Neg for &'a Amplitude {
    type Output = Amplitude;
    fn neg(self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Neg(self.clone())))),
            dependencies: self.clone().dependencies,
            ..Default::default()
        }
    }
}
impl Neg for Amplitude {
    type Output = Amplitude;
    fn neg(self) -> Self::Output {
        Amplitude {
            op: Some(Arc::new(RwLock::new(Operation::Neg(self.clone())))),
            dependencies: self.clone().dependencies,
            ..Default::default()
        }
    }
}

impl From<f64> for Amplitude {
    fn from(value: f64) -> Self {
        Amplitude::new(value.to_string(), move |_, _| Ok(value.into()))
    }
}

impl From<Complex64> for Amplitude {
    fn from(value: Complex64) -> Self {
        Amplitude::new(value.to_string(), move |_, _| Ok(value))
    }
}

pub trait AmplitudeBuilder {
    fn internal_parameter_names(&self) -> Option<Vec<String>>;
    fn to_amplitude(self) -> Amplitude;
    fn with(self, parameters: &[ParameterType]) -> Amplitude
    where
        Self: Sized,
    {
        if let Some(internal_names) = self.internal_parameter_names() {
            println!("Parnames {:?}", internal_names);
            let external_names: Vec<&String> = parameters
                .iter()
                .map(|par_type| match par_type {
                    ParameterType::Scalar(par) => &par.name,
                    ParameterType::CScalar(par) => &par.name,
                })
                .collect();
            internal_names.iter().zip(external_names.iter()).fold(
                self.to_amplitude().with_pars(internal_names.clone()),
                |amp, (i_name, e_name)| amp.map(e_name.to_string(), i_name.to_string()),
            )
        } else {
            self.to_amplitude()
        }
    }
}

#[derive(Clone)]
pub enum ParameterType {
    Scalar(Parameter),
    CScalar(ComplexParameter),
}

impl ParameterType {
    pub fn scalar(&self) -> Result<f64, &str> {
        if let Self::Scalar(par) = self {
            Ok(par.value())
        } else {
            Err("Could not convert to Scalar type")
        }
    }
    pub fn cscalar(&self) -> Result<Complex64, &str> {
        if let Self::CScalar(par) = self {
            Ok(par.value())
        } else {
            Err("Could not convert to CScalar type")
        }
    }
}

impl From<Parameter> for ParameterType {
    fn from(value: Parameter) -> Self {
        ParameterType::Scalar(value)
    }
}

impl From<ComplexParameter> for ParameterType {
    fn from(value: ComplexParameter) -> Self {
        ParameterType::CScalar(value)
    }
}

#[derive(Clone)]
pub struct Parameter {
    name: String,
    pub value: f64,
    lower_bound: Option<f64>,
    upper_bound: Option<f64>,
    fixed: bool,
}

impl Parameter {
    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn new(name: String, value: f64) -> Parameter {
        Parameter {
            name,
            value,
            lower_bound: None,
            upper_bound: None,
            fixed: false,
        }
    }
    pub fn with_bounds(mut self, lower_bound: Option<f64>, upper_bound: Option<f64>) -> Parameter {
        self.lower_bound = lower_bound;
        self.upper_bound = upper_bound;
        self
    }
    pub fn fix(mut self) -> Parameter {
        self.fixed = true;
        self
    }
    pub fn free(mut self) -> Parameter {
        self.fixed = false;
        self
    }
    pub fn as_amp(&self) -> Amplitude {
        Amplitude::new(self.name.clone(), |pars: &ParMap, _vars: &VarMap| {
            Ok(pars["parameter"].scalar()?.into())
        })
        .with_pars(vec!["parameter".to_string()])
        .link(self.name.clone(), "parameter".to_string())
    }
}

#[derive(Clone)]
pub enum Coordinates {
    Cartesian,
    Polar,
}

#[derive(Clone)]
pub struct ComplexParameter {
    pub name: String,
    pub a: Parameter,
    pub b: Parameter,
    pub coordinates: Coordinates,
}

impl ComplexParameter {
    pub fn value(&self) -> Complex64 {
        match self.coordinates {
            Coordinates::Cartesian => Complex64::new(self.a.value(), self.b.value()),
            Coordinates::Polar => Complex64::from_polar(self.a.value(), self.b.value()),
        }
    }

    pub fn as_amp(&self) -> Amplitude {
        let name = self.name.clone();
        Amplitude::new(name.clone(), move |pars: &ParMap, _vars: &VarMap| {
            Ok(pars[&name].cscalar()?)
        })
        .with_pars(vec!["parameter".to_string()])
        .link(self.name.clone(), "parameter".to_string())
    }
}

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
    pub fn new<F>(name: String, function: F, dependencies: Option<Vec<Variable>>) -> Self
    where
        F: 'static + Fn(&VarMap) -> FieldType + Sync + Send,
    {
        Variable {
            name: Arc::new(name),
            function: Arc::new(RwLock::new(function)),
            dependencies,
        }
    }
}

pub trait VariableBuilder {
    fn to_variable(self) -> Variable;
}
