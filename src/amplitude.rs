use argmin::core::CostFunction;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::{
    borrow::BorrowMut,
    cell::RefCell,
    collections::hash_map::Entry,
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
    amplitude: Box<dyn Node>,
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
    pub fn new<A: Node + 'static>(name: &str, amplitude: A) -> Self {
        Self {
            name: name.to_string(),
            amplitude: Box::new(amplitude),
            aux_data: Vec::default(),
        }
    }
    pub fn scalar(name: &str) -> Self {
        Self {
            name: name.to_string(),
            amplitude: Box::new(Scalar),
            aux_data: Vec::default(),
        }
    }
    pub fn cscalar(name: &str) -> Self {
        Self {
            name: name.to_string(),
            amplitude: Box::new(ComplexScalar),
            aux_data: Vec::default(),
        }
    }
    pub fn precompute(&mut self, dataset: &Dataset) {
        self.aux_data = dataset
            .par_iter()
            .map(|event| self.amplitude.precalculate(event))
            .collect();
    }
    pub fn compute(&self, parameters: &Vec<f64>, index: usize, dataset: &Dataset) -> Complex64 {
        self.amplitude
            .calculate(parameters, &dataset.events[index], &self.aux_data[index])
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
    pub sums: FxHashMap<String, FxHashMap<String, Vec<Arc<Amplitude>>>>,
    data: &'d Dataset,
}

impl<'d> Manager<'d> {
    pub fn new(dataset: &'d Dataset) -> Self {
        Self {
            sums: FxHashMap::default(),
            data: dataset,
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
                let mut sum_map = FxHashMap::default();
                sum_map.insert(group_name.to_string(), vec![Arc::clone(amplitude)]);
                sum_map
            });
    }
    fn _compute(
        &self,
        parameters: &Vec<Vec<Vec<Vec<f64>>>>,
        index: usize,
        dataset: &Dataset,
    ) -> f64 {
        // TODO: needs a fix, I forgot that hashmap values aren't ordered...
        let mut res = 0.0;
        for ((sum_name, sum), sum_parameters) in self.sums.iter().zip(parameters) {
            println!("In sum: {sum_name}");
            let mut sum_tot = Complex64::new(0.0, 0.0);
            for ((term_name, term), term_parameters) in sum.iter().zip(sum_parameters) {
                println!("In term: {term_name}");
                let mut term_tot = Complex64::new(1.0, 0.0);
                for (amp, amp_parameters) in term.iter().zip(term_parameters) {
                    println!("Evaluating: {}", amp.name);
                    term_tot *= amp.compute(amp_parameters, index, dataset);
                }
                sum_tot += term_tot;
            }
            res += sum_tot.norm_sqr();
        }
        res
        // self.sums
        //     .values()
        //     .zip(parameters)
        //     .map(|(sum, sum_parameters)| {
        //         sum.values()
        //             .zip(sum_parameters)
        //             .map(|(term, term_parameters)| {
        //                 term.iter()
        //                     .zip(term_parameters)
        //                     .map(|(arcnode, amp_parameters)| {
        //                         arcnode.compute(amp_parameters, index, dataset)
        //                     })
        //                     .product::<Complex64>()
        //             })
        //             .sum::<Complex64>()
        //             .norm_sqr()
        //     })
        //     .sum()
    }
    pub fn compute(&self, parameters: &Vec<Vec<Vec<Vec<f64>>>>) -> Vec<f64> {
        (0..self.data.len())
            .into_iter()
            .map(|index| self._compute(parameters, index, self.data))
            .collect()
    }
}

impl<'d> CostFunction for Manager<'d> {
    type Param = Vec<Vec<Vec<Vec<f64>>>>;
    type Output = f64;
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin_math::Error> {
        Ok(self.compute(param).iter().sum())
    }
}
