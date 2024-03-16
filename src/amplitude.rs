use crate::{dataset::Dataset, prelude::Event};
use num_complex::Complex64;
use rustc_hash::FxHashMap as HashMap;

#[derive(Default)]
pub struct ParameterSet(HashMap<String, f64>);
impl ParameterSet {
    pub fn get(&self, parameter: &str) -> f64 {
        match self.0.get(parameter) {
            Some(val) => *val,
            None => panic!("Missing parameter {}", parameter),
        }
    }
    pub fn insert(&mut self, name: &str, value: f64) {
        self.0.insert(name.to_string(), value);
    }
}

pub trait Amplitude<T: Event> {
    fn calculate(&mut self, parameters: &ParameterSet, dataset: &Dataset<T>) -> Vec<Complex64>;
}
