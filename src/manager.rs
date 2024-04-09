use gomez::Function;
use gomez::Problem;
use indexmap::IndexMap as OHashMap;
use num_complex::Complex64;
use rayon::prelude::*;
use std::{
    cmp::Ordering,
    sync::{Arc, RwLock},
};

use crate::prelude::{Amplitude, Dataset, Event};

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

/// An enum which stores an [`Amplitude`] in an activated/deactivated state.
///
/// This enum contains two subtypes which just describe whether an [`Amplitude`] is activated or
/// deactivated within the context of a [`Manager`]. Activated amplitudes are included in the total
/// calculation, while deactivated amplitudes are ignored. This has no impact on which parameters
/// are included in the input set, so code which runs on a [`Manager`] will run regardless if a
/// particular [`Amplitude`] is activated or deactivated. This can be used to disable an
/// [`Amplitude`] for calculating a partial projection from a fit. For instance, if a fit contains
/// multiple Gaussians, deactivating all but one will isolate a particular Gaussian.
#[derive(Debug)]
pub enum AmplitudeType {
    /// Indicates an [`Amplitude`] is in the "activated" state.
    Activated(Arc<RwLock<Amplitude>>),

    /// Indicates an [`Amplitude`] is in the "deactivated" state.
    Deactivated(Arc<RwLock<Amplitude>>),
}
impl AmplitudeType {
    pub fn is_activated(&self) -> bool {
        //! Returns `true` if the [`Amplitude`] is activated, otherwise `false`.
        match self {
            Self::Activated(_) => true,
            Self::Deactivated(_) => false,
        }
    }
    pub fn activate(&mut self) {
        //! Converts an [`AmplitudeType::Deactivated`] type into an [`AmplitudeType::Activated`]
        //! type.
        if let Self::Deactivated(ref arc) = self {
            *self = Self::Activated(Arc::clone(arc));
        }
    }
    pub fn deactivate(&mut self) {
        //! Converts an [`AmplitudeType::Activated`] type into an [`AmplitudeType::Deactivated`]
        //! type.
        if let Self::Activated(ref arc) = self {
            *self = Self::Deactivated(Arc::clone(arc));
        }
    }
    pub fn get_amplitude(&self) -> &Arc<RwLock<Amplitude>> {
        //! Retrieves the internal [`Amplitude`] as a reference to an [`Arc<RwLock<Amplitude>>`].
        match self {
            Self::Activated(ref arc) => arc,
            Self::Deactivated(ref arc) => arc,
        }
    }
}

pub trait Manage {
    fn parameters(&self) -> Vec<String>;
    fn register(&mut self, sum_name: &str, group_name: &str, amplitude: &Arc<RwLock<Amplitude>>);
    fn precompute(&mut self);
    fn constrain(
        &mut self,
        parameter_1: (&str, &str, &str, &str),
        parameter_2: (&str, &str, &str, &str),
    );
    fn constrain_amplitude(&mut self, group_1: (&str, &str, &str), group_2: (&str, &str, &str));
    fn activate(&mut self, amplitude: (&str, &str, &str));
    fn deactivate(&mut self, amplitude: (&str, &str, &str));
    fn fix(&mut self, parameter: (&str, &str, &str, &str), value: f64);
    fn free(&mut self, parameter: (&str, &str, &str, &str));
}

/// A convenience type used to store a structure of [`Amplitude`]s.
///
/// This type is organized into two nested *ordered* [`indexmap::IndexMap`]s. The outer map
/// describes "sums" while the inner map describes "groups". Each group contains a
/// [`Vec<AmplitudeType>`]. In the total calculation, the [`Amplitude`]s contained in each group
/// are multiplied together, and the groups in each sum are added together. The absolute-square of
/// each sum is then added to form the total.
type SumMap = OHashMap<String, OHashMap<String, Vec<AmplitudeType>>>;

/// A convenience type used to store the structure of parameters.
///
/// This type is organized into three nested *ordered* [`indexmap::IndexMap`]s. The outer map
/// describes "sums" while the middle map describes "groups". The inner map maps [`Amplitude`]s to
/// a named list of their parameters. See [`SumMap`] for more information on this.
type ParMap = OHashMap<String, OHashMap<String, OHashMap<String, Vec<(String, ParameterType)>>>>;

/// A struct to manage a single [`Dataset`] and an arbitrary number of [`Amplitude`]s.
///
/// The [`Manager`] struct stores a reference to a [`Dataset`] and all of the mechanics to actually
/// run calculations over that [`Dataset`]. Every analysis follows the following form:
///
/// ```math
/// I(\overrightarrow{p}, e) = \sum_{\text{sums}}\left|\sum_{\text{groups}} \prod_{\text{amp} \in \text{amplitudes}} \text{amp}(\overrightarrow{p}, e)\right|^2
/// ```
///
/// where $`\overrightarrow{p}`$ is a vector of parameters and $`e`$ represents the data from an
/// [`Event`].
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
        //! Creates a new [`Manager`] from a &[`Dataset`].
        //!
        //! This is the prefered method for creating new [`Manager`]s. Because no modification ever
        //! happens to the [`Dataset`] itself, multiple [`Manager`]s should be able to be made from
        //! the same [`Dataset`]
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
impl<'d> Manage for Manager<'d> {
    fn parameters(&self) -> Vec<String> {
        let mut output: Vec<String> = Vec::with_capacity(self.variable_count);
        for (_, sum) in self.sums.iter() {
            for (_, group) in sum.iter() {
                for amplitude in group.iter() {
                    let params = amplitude.get_amplitude().read().unwrap().node.parameters();
                    if let Some(pars) = params {
                        for par in pars {
                            output.push(par);
                        }
                    }
                }
            }
        }
        output
    }
    fn register(&mut self, sum_name: &str, group_name: &str, amplitude: &Arc<RwLock<Amplitude>>) {
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
    fn activate(&mut self, amplitude: (&str, &str, &str)) {
        let (sum_name, group_name, amplitude_name) = amplitude;
        self.get_amplitudetype_mut(sum_name, group_name, amplitude_name)
            .activate();
    }
    fn deactivate(&mut self, amplitude: (&str, &str, &str)) {
        let (sum_name, group_name, amplitude_name) = amplitude;
        self.get_amplitudetype_mut(sum_name, group_name, amplitude_name)
            .deactivate();
    }
    fn fix(&mut self, parameter: (&str, &str, &str, &str), value: f64) {
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
    fn free(&mut self, parameter: (&str, &str, &str, &str)) {
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
    fn constrain(
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
    fn constrain_amplitude(&mut self, group_1: (&str, &str, &str), group_2: (&str, &str, &str)) {
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
    fn precompute(&mut self) {
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
}
impl<'a> Manage for MultiManager<'a> {
    fn parameters(&self) -> Vec<String> {
        self.managers[0].parameters()
    }
    fn register(&mut self, sum_name: &str, group_name: &str, amplitude: &Arc<RwLock<Amplitude>>) {
        self.managers.iter_mut().for_each(|manager| {
            let amp = (*amplitude).clone(); // TODO: This doesn't actually work.
            manager.register(sum_name, group_name, &amp);
        });
    }
    fn activate(&mut self, amplitude: (&str, &str, &str)) {
        self.managers.iter_mut().for_each(|manager| {
            manager.activate(amplitude);
        });
    }
    fn deactivate(&mut self, amplitude: (&str, &str, &str)) {
        self.managers.iter_mut().for_each(|manager| {
            manager.deactivate(amplitude);
        });
    }
    fn fix(&mut self, parameter: (&str, &str, &str, &str), value: f64) {
        self.managers.iter_mut().for_each(|manager| {
            manager.fix(parameter, value);
        });
    }
    fn free(&mut self, parameter: (&str, &str, &str, &str)) {
        self.managers.iter_mut().for_each(|manager| {
            manager.free(parameter);
        });
    }
    fn constrain(
        &mut self,
        parameter_1: (&str, &str, &str, &str),
        parameter_2: (&str, &str, &str, &str),
    ) {
        self.managers.iter_mut().for_each(|manager| {
            manager.constrain(parameter_1, parameter_2);
        });
    }
    fn constrain_amplitude(&mut self, group_1: (&str, &str, &str), group_2: (&str, &str, &str)) {
        self.managers.iter_mut().for_each(|manager| {
            manager.constrain_amplitude(group_1, group_2);
        });
    }
    fn precompute(&mut self) {
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
        let data_result: f64 = self.manager.managers[0]
            .compute(parameters)
            .iter()
            .zip(self.manager.managers[0].data.iter())
            .map(|(res, e)| e.weight * res.ln())
            .sum();
        let mc_result: f64 = self.manager.managers[1]
            .compute(parameters)
            .iter()
            .zip(self.manager.managers[1].data.iter())
            .map(|(res, e)| e.weight * res)
            .sum();
        let n_data = self.manager.managers[0].data.len() as f64;
        let n_mc = self.manager.managers[1].data.len() as f64;
        data_result - (n_data / n_mc) * mc_result
    }
}
impl<'a> Manage for ExtendedLogLikelihood<'a> {
    fn parameters(&self) -> Vec<String> {
        self.manager.parameters()
    }
    fn register(&mut self, sum_name: &str, group_name: &str, amplitude: &Arc<RwLock<Amplitude>>) {
        self.manager.register(sum_name, group_name, amplitude);
    }
    fn precompute(&mut self) {
        self.manager.precompute();
    }
    fn constrain(
        &mut self,
        parameter_1: (&str, &str, &str, &str),
        parameter_2: (&str, &str, &str, &str),
    ) {
        self.manager.constrain(parameter_1, parameter_2);
    }
    fn constrain_amplitude(&mut self, group_1: (&str, &str, &str), group_2: (&str, &str, &str)) {
        self.manager.constrain_amplitude(group_1, group_2);
    }
    fn activate(&mut self, amplitude: (&str, &str, &str)) {
        self.manager.activate(amplitude);
    }
    fn deactivate(&mut self, amplitude: (&str, &str, &str)) {
        self.manager.deactivate(amplitude);
    }
    fn fix(&mut self, parameter: (&str, &str, &str, &str), value: f64) {
        self.manager.fix(parameter, value);
    }
    fn free(&mut self, parameter: (&str, &str, &str, &str)) {
        self.manager.free(parameter);
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
