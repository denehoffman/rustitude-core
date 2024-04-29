use indexmap::IndexMap as OHashMap;
use num_complex::Complex64;
use rayon::prelude::*;
use std::{cmp::Ordering, fmt::Display};

use crate::prelude::{Amplitude, Dataset, Event};

/// A struct which specifies if a given parameter is fixed or free and gives the corresponding value
/// and input index as well as information about the parameter's position in sums, groups, and
/// amplitudes.
///
/// This struct is mostly used internally by the [`Manager`] struct. The index [`usize`] corresponds
/// to the index this parameter should be sourced from in the input vector. [`Parameter`]s will have
/// input indices larger than the number of free parameters in the system, and parameter indices
/// will automatically be incremented and decremented when new parameters are added, fixed, or
/// constrained.
///
/// See also: [`Manager::fix`], [`Manager::free`], [`Manager::constrain`]
#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub struct Parameter {
    sum: String,
    group: String,
    amplitude: String,
    name: String,
    fixed: Option<f64>,
    index: usize,
    initial_value: f64,
    lower_bound: f64,
    upper_bound: f64,
}

impl Display for Parameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -> ({}:::{}:::{}) {} {}",
            self.index,
            self.sum,
            self.group,
            self.amplitude,
            self.name,
            if let Some(value) = self.fixed {
                format!("({value})")
            } else {
                "".to_string()
            }
        )
    }
}

impl Parameter {
    pub fn new(sum: &str, group: &str, amplitude: &str, name: &str, index: usize) -> Self {
        Self {
            sum: sum.to_string(),
            group: group.to_string(),
            amplitude: amplitude.to_string(),
            name: name.to_string(),
            index,
            fixed: None,
            initial_value: 0.0,
            lower_bound: f64::NEG_INFINITY,
            upper_bound: f64::INFINITY,
        }
    }
    pub fn increment(&mut self) {
        //! Increments the index by `1`.
        self.index += 1;
    }
    pub fn decrement(&mut self) {
        //! Decrements the index by `1`.
        self.index -= 1;
    }
    pub fn set_index(&mut self, index: usize) {
        //! Sets the index to a given value.
        self.index = index;
    }
    pub fn get_index(&self) -> usize {
        //! Getter method for the index.
        self.index
    }
    pub fn fix(&mut self, index: usize, value: f64) {
        //! Converts the [`Parameter`] from free to fixed with the given index and value.
        self.fixed = Some(value);
        self.index = index;
        self.initial_value = value;
    }
    pub fn free(&mut self, index: usize, initial_value: f64) {
        //! Converts the [`Parameter`] from fixed to free with a given index and initial
        //! value.
        self.fixed = None;
        self.index = index;
        self.initial_value = initial_value;
    }
    pub fn is_fixed(&self) -> bool {
        //! Checks if the [`Parameter`] is fixed.
        self.fixed.is_some()
    }
    pub fn fixed(&self) -> Option<f64> {
        //! Getter method for fixed.
        self.fixed
    }
    pub fn set_bounds(&mut self, lower_bound: f64, upper_bound: f64) {
        self.lower_bound = lower_bound;
        self.upper_bound = upper_bound;
    }
    pub fn set_initial(&mut self, initial_value: f64) {
        self.initial_value = initial_value;
    }
    pub fn get_bounds(&self) -> (f64, f64) {
        (self.lower_bound, self.upper_bound)
    }
    pub fn get_initial(&self) -> f64 {
        self.initial_value
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
    Activated(Amplitude),

    /// Indicates an [`Amplitude`] is in the "deactivated" state.
    Deactivated(Amplitude),
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
        if let Self::Deactivated(ref amp) = self {
            *self = Self::Activated(amp.clone()); // TODO: fix this?
        }
    }
    pub fn deactivate(&mut self) {
        //! Converts an [`AmplitudeType::Activated`] type into an [`AmplitudeType::Deactivated`]
        //! type.
        if let Self::Activated(ref arc) = self {
            *self = Self::Deactivated(arc.clone());
        }
    }
    pub fn get_amplitude(&self) -> &Amplitude {
        //! Retrieves the internal [`Amplitude`] as a reference to an [`Arc<RwLock<Amplitude>>`].
        match self {
            Self::Activated(ref arc) => arc,
            Self::Deactivated(ref arc) => arc,
        }
    }
}

type AmplitudeID<'a> = (&'a str, &'a str, &'a str);
type ParameterID<'a> = (&'a str, &'a str, &'a str, &'a str);

pub trait Manage {
    fn parameters(&self) -> Vec<Parameter>;
    fn register(&mut self, sum_name: &str, group_name: &str, amplitude: &Amplitude);
    fn precompute(&mut self);
    fn constrain(&mut self, parameter_1: ParameterID, parameter_2: ParameterID);
    fn constrain_amplitude(&mut self, amplitude_1: AmplitudeID, amplitude_2: AmplitudeID);
    fn activate(&mut self, amplitude: AmplitudeID);
    fn deactivate(&mut self, amplitude: AmplitudeID);
    fn fix(&mut self, parameter: ParameterID, value: f64);
    fn free(&mut self, parameter: ParameterID, initial_value: f64);
    fn set_bounds(&mut self, parameter: ParameterID, lower_bound: f64, upper_bound: f64);
    fn set_initial(&mut self, parameter: ParameterID, initial_value: f64);
    fn get_lower_bounds(&self) -> Vec<f64> {
        self.parameters().iter().map(|p| p.lower_bound).collect()
    }
    fn get_upper_bounds(&self) -> Vec<f64> {
        self.parameters().iter().map(|p| p.upper_bound).collect()
    }
    fn get_initial_values(&self) -> Vec<f64> {
        self.parameters().iter().map(|p| p.initial_value).collect()
    }
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
type ParMap = OHashMap<String, OHashMap<String, OHashMap<String, Vec<(String, Parameter)>>>>;

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
pub struct Manager {
    pub sums: SumMap,
    pub pars: ParMap,
    data: Dataset,
    variable_count: usize,
    fixed_variable_count: usize,
}

impl Manager {
    pub fn new(dataset: &Dataset) -> Self {
        //! Creates a new [`Manager`] from a &[`Dataset`].
        //!
        //! This is the prefered method for creating new [`Manager`]s. Because no modification ever
        //! happens to the [`Dataset`] itself, multiple [`Manager`]s should be able to be made from
        //! the same [`Dataset`]
        Self {
            sums: SumMap::default(),
            pars: ParMap::default(),
            data: dataset.clone(),
            variable_count: 0,
            fixed_variable_count: 0,
        }
    }
    fn get_parameter(
        &self,
        sum_name: &str,
        group_name: &str,
        amplitude_name: &str,
        parameter_name: &str,
    ) -> &Parameter {
        self.pars
            .get(sum_name)
            .unwrap_or_else(|| panic!("Could not find {}", sum_name))
            .get(group_name)
            .unwrap_or_else(|| panic!("Could not find {}", group_name))
            .get(amplitude_name)
            .unwrap_or_else(|| panic!("Could not find {}", amplitude_name))
            .iter()
            .find(|(par_name, _)| *par_name == parameter_name)
            .map(|(_, parameter)| parameter)
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
            .find(|amplitude_type| amplitude_type.get_amplitude().name == amplitude_name)
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
            .find(|amplitude_type| amplitude_type.get_amplitude().name == amplitude_name)
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
    fn apply_to_parameters(&mut self, closure: impl Fn(&mut Parameter)) {
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
                                    .map(|param| match param.1.fixed() {
                                        Some(val) => val,
                                        None => parameters[param.1.get_index()],
                                    })
                                    .collect();
                                if amplitude_type.is_activated() {
                                    amplitude_type
                                        .get_amplitude()
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
            .events
            .read()
            .par_iter()
            .map(|event| self._compute(parameters, event))
            .collect()
    }
}
impl Manage for Manager {
    fn parameters(&self) -> Vec<Parameter> {
        let mut output: Vec<Parameter> = Vec::with_capacity(self.variable_count);
        for (_, sum) in self.pars.iter() {
            for (_, group) in sum.iter() {
                for (_, amplitude) in group.iter() {
                    for (_, parameter) in amplitude.iter() {
                        output.push(parameter.clone());
                    }
                }
            }
        }
        output
    }
    fn register(&mut self, sum_name: &str, group_name: &str, amplitude: &Amplitude) {
        amplitude.precompute(&self.data);
        let amp_name = amplitude.name.clone();
        self.sums
            .entry(sum_name.to_string())
            .and_modify(|sum_entry| {
                sum_entry
                    .entry(group_name.to_string())
                    .and_modify(|group_entry| {
                        group_entry.push(AmplitudeType::Activated(amplitude.clone()))
                    })
                    .or_insert(vec![AmplitudeType::Activated(amplitude.clone())]);
            })
            .or_insert_with(|| {
                let mut sum_map = OHashMap::default();
                sum_map.insert(
                    group_name.to_string(),
                    vec![AmplitudeType::Activated(amplitude.clone())],
                );
                sum_map
            });
        let mut pars: Vec<(String, Parameter)> = Vec::new();
        if let Some(parameter_names) = amplitude.node.read().parameters() {
            for parameter_name in parameter_names {
                pars.push((
                    parameter_name.clone(),
                    Parameter::new(
                        sum_name,
                        group_name,
                        &amp_name,
                        &parameter_name.clone(),
                        self.variable_count,
                    ),
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
    fn activate(&mut self, amplitude: AmplitudeID) {
        let (sum_name, group_name, amplitude_name) = amplitude;
        self.get_amplitudetype_mut(sum_name, group_name, amplitude_name)
            .activate();
    }
    fn deactivate(&mut self, amplitude: AmplitudeID) {
        let (sum_name, group_name, amplitude_name) = amplitude;
        self.get_amplitudetype_mut(sum_name, group_name, amplitude_name)
            .deactivate();
    }
    fn fix(&mut self, parameter: ParameterID, value: f64) {
        let (sum_name, group_name, amplitude_name, parameter_name) = parameter;
        let parameter = self
            .get_parameter(sum_name, group_name, amplitude_name, parameter_name)
            .clone();
        let mut new_parameter = parameter.clone();
        new_parameter.fix(self.variable_count + self.fixed_variable_count, value);
        self.apply_to_parameters(|other| {
            if other.get_index() == parameter.get_index() {
                *other = new_parameter.clone();
            }
            if other.get_index() > parameter.get_index() {
                other.decrement();
            }
        });
        self.fixed_variable_count += 1;
        self.variable_count -= 1;
    }
    fn free(&mut self, parameter: ParameterID, initial_value: f64) {
        let (sum_name, group_name, amplitude_name, parameter_name) = parameter;
        let parameter = self
            .get_parameter(sum_name, group_name, amplitude_name, parameter_name)
            .clone();
        let mut new_parameter = parameter.clone();
        new_parameter.free(self.variable_count, initial_value);
        self.apply_to_parameters(
            |other| match other.get_index().cmp(&parameter.get_index()) {
                Ordering::Less => {
                    if other.get_index() >= new_parameter.get_index() {
                        other.increment();
                    }
                }
                Ordering::Equal => *other = new_parameter.clone(),
                Ordering::Greater => {}
            },
        );
        self.fixed_variable_count -= 1;
        self.variable_count += 1;
    }
    fn constrain(&mut self, parameter_1: ParameterID, parameter_2: ParameterID) {
        let (sum_name_1, group_name_1, amplitude_name_1, parameter_name_1) = parameter_1;
        let (sum_name_2, group_name_2, amplitude_name_2, parameter_name_2) = parameter_2;
        let parameter_1 = self
            .get_parameter(sum_name_1, group_name_1, amplitude_name_1, parameter_name_1)
            .clone();
        let parameter_2 = self
            .get_parameter(sum_name_2, group_name_2, amplitude_name_2, parameter_name_2)
            .clone();
        let index_1 = parameter_1.get_index();
        let index_2 = parameter_2.get_index();
        self.apply_to_parameters(|parameter| {
            let par_index = parameter.get_index();
            if index_1 > index_2 {
                if par_index == index_1 {
                    *parameter = parameter_2.clone();
                }
                if par_index > index_1 {
                    parameter.decrement();
                }
            }
            if index_1 < index_2 {
                if par_index == index_2 {
                    *parameter = parameter_1.clone();
                }
                if par_index > index_2 {
                    parameter.decrement();
                }
            }
        });
        self.variable_count -= 1;
    }
    fn constrain_amplitude(&mut self, amplitude_1: AmplitudeID, amplitude_2: AmplitudeID) {
        let (sum_name_1, group_name_1, amplitude_name_1) = amplitude_1;
        let (sum_name_2, group_name_2, amplitude_name_2) = amplitude_2;
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
                    amplitude.get_amplitude().precompute(&self.data)
                }
            }
        }
    }

    fn set_bounds(&mut self, parameter: ParameterID, lower_bound: f64, upper_bound: f64) {
        let (sum_name, group_name, amplitude_name, parameter_name) = parameter;
        let parameter = self
            .get_parameter(sum_name, group_name, amplitude_name, parameter_name)
            .clone();
        let mut new_parameter = parameter.clone();
        new_parameter.set_bounds(lower_bound, upper_bound);
        self.apply_to_parameters(|other| {
            if other.get_index() == parameter.get_index() {
                *other = new_parameter.clone();
            }
        });
    }

    fn set_initial(&mut self, parameter: ParameterID, initial_value: f64) {
        let (sum_name, group_name, amplitude_name, parameter_name) = parameter;
        let parameter = self
            .get_parameter(sum_name, group_name, amplitude_name, parameter_name)
            .clone();
        let mut new_parameter = parameter.clone();
        new_parameter.set_initial(initial_value);
        self.apply_to_parameters(|other| {
            if other.get_index() == parameter.get_index() {
                *other = new_parameter.clone();
            }
        });
    }
}

pub struct MultiManager {
    managers: Vec<Manager>,
}

impl MultiManager {
    pub fn new(datasets: Vec<&Dataset>) -> Self {
        Self {
            managers: datasets.iter().map(|ds| Manager::new(ds)).collect(),
        }
    }
}
impl Manage for MultiManager {
    fn parameters(&self) -> Vec<Parameter> {
        self.managers[0].parameters()
    }
    fn register(&mut self, sum_name: &str, group_name: &str, amplitude: &Amplitude) {
        self.managers.iter_mut().for_each(|manager| {
            let amp = (*amplitude).clone(); // TODO: This doesn't actually work.
            manager.register(sum_name, group_name, &amp);
        });
    }
    fn activate(&mut self, amplitude: AmplitudeID) {
        self.managers.iter_mut().for_each(|manager| {
            manager.activate(amplitude);
        });
    }
    fn deactivate(&mut self, amplitude: AmplitudeID) {
        self.managers.iter_mut().for_each(|manager| {
            manager.deactivate(amplitude);
        });
    }
    fn fix(&mut self, parameter: ParameterID, value: f64) {
        self.managers.iter_mut().for_each(|manager| {
            manager.fix(parameter, value);
        });
    }
    fn free(&mut self, parameter: ParameterID, initial_value: f64) {
        self.managers.iter_mut().for_each(|manager| {
            manager.free(parameter, initial_value);
        });
    }
    fn constrain(&mut self, parameter_1: ParameterID, parameter_2: ParameterID) {
        self.managers.iter_mut().for_each(|manager| {
            manager.constrain(parameter_1, parameter_2);
        });
    }
    fn constrain_amplitude(&mut self, amplitude_1: AmplitudeID, amplitude_2: AmplitudeID) {
        self.managers.iter_mut().for_each(|manager| {
            manager.constrain_amplitude(amplitude_1, amplitude_2);
        });
    }
    fn precompute(&mut self) {
        self.managers.iter_mut().for_each(|manager| {
            manager.precompute();
        });
    }

    fn set_bounds(&mut self, parameter: ParameterID, lower_bound: f64, upper_bound: f64) {
        self.managers.iter_mut().for_each(|manager| {
            manager.set_bounds(parameter, lower_bound, upper_bound);
        });
    }

    fn set_initial(&mut self, parameter: ParameterID, initial_value: f64) {
        self.managers.iter_mut().for_each(|manager| {
            manager.set_initial(parameter, initial_value);
        });
    }
}

pub struct ExtendedLogLikelihood {
    pub manager: MultiManager,
}
impl ExtendedLogLikelihood {
    pub fn new(data: &Dataset, monte_carlo: &Dataset) -> Self {
        Self {
            manager: MultiManager::new(vec![data, monte_carlo]),
        }
    }
    pub fn compute(&self, parameters: &[f64]) -> f64 {
        let data_result: f64 = self.manager.managers[0]
            .compute(parameters)
            .iter()
            .zip(self.manager.managers[0].data.events.read().iter())
            .map(|(res, e)| e.weight * res.ln())
            .sum();
        let mc_result: f64 = self.manager.managers[1]
            .compute(parameters)
            .iter()
            .zip(self.manager.managers[1].data.events.read().iter())
            .map(|(res, e)| e.weight * res)
            .sum();
        let n_data = self.manager.managers[0].data.len() as f64;
        let n_mc = self.manager.managers[1].data.len() as f64;
        data_result - (n_data / n_mc) * mc_result
    }
}
impl Manage for ExtendedLogLikelihood {
    fn parameters(&self) -> Vec<Parameter> {
        self.manager.parameters()
    }
    fn register(&mut self, sum_name: &str, group_name: &str, amplitude: &Amplitude) {
        self.manager.register(sum_name, group_name, amplitude);
    }
    fn precompute(&mut self) {
        self.manager.precompute();
    }
    fn constrain(&mut self, parameter_1: ParameterID, parameter_2: ParameterID) {
        self.manager.constrain(parameter_1, parameter_2);
    }
    fn constrain_amplitude(&mut self, amplitude_1: AmplitudeID, amplitude_2: AmplitudeID) {
        self.manager.constrain_amplitude(amplitude_1, amplitude_2);
    }
    fn activate(&mut self, amplitude: AmplitudeID) {
        self.manager.activate(amplitude);
    }
    fn deactivate(&mut self, amplitude: AmplitudeID) {
        self.manager.deactivate(amplitude);
    }
    fn fix(&mut self, parameter: ParameterID, value: f64) {
        self.manager.fix(parameter, value);
    }
    fn free(&mut self, parameter: ParameterID, initial_value: f64) {
        self.manager.free(parameter, initial_value);
    }

    fn set_bounds(&mut self, parameter: ParameterID, lower_bound: f64, upper_bound: f64) {
        self.manager.set_bounds(parameter, lower_bound, upper_bound);
    }

    fn set_initial(&mut self, parameter: ParameterID, initial_value: f64) {
        self.manager.set_initial(parameter, initial_value);
    }
}
