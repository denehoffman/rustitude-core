use argmin::core::CostFunction;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::{
    borrow::BorrowMut,
    cell::RefCell,
    collections::{hash_map::Entry, BTreeMap, BTreeSet},
    fmt::Debug,
    sync::{Arc, Mutex},
};

use num_complex::Complex64;

use crate::{dataset::Event, prelude::Dataset};

pub trait Node: Sync + Send {
    fn precalculate(&self, event: &Event) -> Vec<f64>;
    fn calculate(&self, parameters: &Vec<f64>, event: &Event, aux_data: &Vec<f64>) -> Complex64;
    fn parameters(&self) -> Option<Vec<String>>;
}

pub struct Amplitude {
    name: String,
    node: Box<dyn Node>,
    aux_data: Vec<Vec<f64>>,
}
impl Debug for Amplitude {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.name)?;
        if self.aux_data.len() >= 5 {
            writeln!(f, "{:?}", &self.aux_data[0..5])?;
        }
        Ok(())
    }
}
impl Amplitude {
    pub fn new<N: Node + 'static>(name: &str, node: N) -> Self {
        Self {
            name: name.to_string(),
            node: Box::new(node),
            aux_data: Vec::default(),
        }
    }
    pub fn scalar(name: &str) -> Self {
        Self {
            name: name.to_string(),
            node: Box::new(Scalar),
            aux_data: Vec::default(),
        }
    }
    pub fn cscalar(name: &str) -> Self {
        Self {
            name: name.to_string(),
            node: Box::new(ComplexScalar),
            aux_data: Vec::default(),
        }
    }
    pub fn precompute(&mut self, dataset: &Dataset) {
        self.aux_data = dataset
            .par_iter()
            .map(|event| self.node.precalculate(event))
            .collect();
    }
    pub fn compute(&self, parameters: &Vec<f64>, index: usize, dataset: &Dataset) -> Complex64 {
        if self.aux_data.len() == 0 {
            self.node
                .calculate(parameters, &dataset.events[index], &vec![])
        } else {
            self.node
                .calculate(parameters, &dataset.events[index], &self.aux_data[index])
        }
    }
}

pub struct Scalar;
impl Node for Scalar {
    fn precalculate(&self, _event: &Event) -> Vec<f64> {
        Vec::default()
    }
    fn calculate(&self, parameters: &Vec<f64>, _event: &Event, _aux_data: &Vec<f64>) -> Complex64 {
        parameters[0].into()
    }
    fn parameters(&self) -> Option<Vec<String>> {
        Some(vec!["value".to_string()])
    }
}
pub struct ComplexScalar;
impl Node for ComplexScalar {
    fn precalculate(&self, _event: &Event) -> Vec<f64> {
        Vec::default()
    }
    fn calculate(&self, parameters: &Vec<f64>, _event: &Event, _aux_data: &Vec<f64>) -> Complex64 {
        Complex64::new(parameters[0], parameters[1])
    }
    fn parameters(&self) -> Option<Vec<String>> {
        Some(vec!["real".to_string(), "imag".to_string()])
    }
}

pub struct Parameter(String, f64);

pub struct Manager<'d> {
    pub sums: BTreeMap<String, BTreeMap<String, Vec<Arc<Amplitude>>>>,
    pub pars: BTreeMap<String, BTreeMap<String, Vec<Vec<(String, usize)>>>>,
    data: &'d Dataset,
    variable_count: usize,
}

impl<'d> Manager<'d> {
    pub fn new(dataset: &'d Dataset) -> Self {
        Self {
            sums: BTreeMap::default(),
            pars: BTreeMap::default(),
            data: dataset,
            variable_count: 0,
        }
    }
    pub fn register(&mut self, sum_name: &str, group_name: &str, amplitude: &Arc<Amplitude>) {
        self.sums
            .entry(sum_name.to_string())
            .and_modify(|sum_entry| {
                sum_entry
                    .entry(group_name.to_string())
                    .and_modify(|group_entry| group_entry.push(Arc::clone(amplitude)))
                    .or_insert(vec![Arc::clone(amplitude)]);
            })
            .or_insert_with(|| {
                let mut sum_map = BTreeMap::default();
                sum_map.insert(group_name.to_string(), vec![Arc::clone(amplitude)]);
                sum_map
            });
        let mut pars: Vec<(String, usize)> = Vec::new();
        if let Some(parameter_names) = amplitude.node.parameters() {
            for parameter_name in parameter_names {
                pars.push((
                    format!("{}::{}", amplitude.name, parameter_name.clone()),
                    self.variable_count,
                ));
                self.variable_count += 1;
            }
        }
        self.pars
            .entry(sum_name.to_string())
            .and_modify(|sum_entry| {
                sum_entry
                    .entry(group_name.to_string())
                    .and_modify(|group_entry| group_entry.push(pars.clone()))
                    .or_insert(vec![pars.clone()]);
            })
            .or_insert_with(|| {
                let mut sum_map = BTreeMap::default();
                sum_map.insert(group_name.to_string(), vec![pars]);
                sum_map
            });
    }
    fn _compute(&self, parameters: &Vec<f64>, index: usize, dataset: &Dataset) -> f64 {
        self.sums
            .values()
            .zip(self.pars.values())
            .map(|(sum, sum_parameters)| {
                sum.values()
                    .zip(sum_parameters.values())
                    .map(|(term, term_parameters)| {
                        term.iter()
                            .zip(term_parameters)
                            .map(|(arcnode, arcnode_parameters)| {
                                let amp_parameters = &arcnode_parameters
                                    .iter()
                                    .map(|param| parameters[param.1])
                                    .collect();
                                arcnode.compute(amp_parameters, index, dataset)
                            })
                            .product::<Complex64>()
                    })
                    .sum::<Complex64>()
                    .norm_sqr()
            })
            .sum()
    }
    pub fn compute(&self, parameters: Vec<f64>) -> Vec<f64> {
        (0..self.data.len())
            .into_iter()
            .map(|index| self._compute(&parameters, index, self.data))
            .collect()
    }
}

// impl<'d> CostFunction for Manager<'d> {
//     type Param = Vec<Vec<Vec<Vec<f64>>>>;
//     type Output = f64;
//     fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin_math::Error> {
//         Ok(self.compute(param).iter().sum())
//     }
// }
