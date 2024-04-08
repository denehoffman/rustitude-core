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

pub trait Node: Sync + Send {
    fn precalculate(&mut self, dataset: &Dataset);
    fn calculate(&self, parameters: &[f64], event: &Event) -> Complex64;
    fn parameters(&self) -> Option<Vec<String>>;
}

pub struct Amplitude {
    pub name: String,
    pub node: Box<dyn Node>,
}
impl Debug for Amplitude {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.name)?;
        Ok(())
    }
}
impl Amplitude {
    /// Creates an Amplitude from a Node.
    ///
    /// # Examples
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
    /// assert_eq!(dbg!(Amplitude::new("A", A).name), "A".to_string());
    /// ```
    pub fn new<N: Node + 'static>(name: &str, node: N) -> Self {
        Self {
            name: name.to_string(),
            node: Box::new(node),
        }
    }
    pub fn scalar(name: &str) -> Self {
        Self {
            name: name.to_string(),
            node: Box::new(Scalar),
        }
    }
    pub fn cscalar(name: &str) -> Self {
        Self {
            name: name.to_string(),
            node: Box::new(ComplexScalar),
        }
    }
    pub fn precompute(&mut self, dataset: &Dataset) {
        self.node.precalculate(dataset);
    }
    pub fn compute(&self, parameters: &[f64], event: &Event) -> Complex64 {
        self.node.calculate(parameters, event)
    }
}

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

#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
pub enum ParameterType {
    Free(usize),
    Fixed(usize, f64),
}
impl ParameterType {
    pub fn increment(&mut self) {
        match self {
            Self::Free(ref mut ind) => *ind += 1,
            Self::Fixed(ref mut ind, _) => *ind += 1,
        }
    }
    pub fn decrement(&mut self) {
        match self {
            Self::Free(ref mut ind) => *ind -= 1,
            Self::Fixed(ref mut ind, _) => *ind -= 1,
        }
    }
    pub fn set_index(&mut self, index: usize) {
        match self {
            Self::Free(ref mut ind) => *ind = index,
            Self::Fixed(ref mut ind, _) => *ind = index,
        }
    }
    pub fn get_index(&self) -> usize {
        match self {
            Self::Free(ref ind) => *ind,
            Self::Fixed(ref ind, _) => *ind,
        }
    }
    pub fn fix(&mut self, index: usize, value: f64) {
        if let Self::Free(_) = self {
            *self = Self::Fixed(index, value);
        }
    }
    pub fn free(&mut self, index: usize) {
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
