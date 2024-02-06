use std::collections::VecDeque;

use rustc_hash::FxHashMap as HashMap;

use anyinput::anyinput;
use argmin::core::{CostFunction, Error};
use rayon::prelude::*;

use crate::{dataset::Dataset, node::Node};

#[macro_export]
macro_rules! par {
    ($name:expr, $value:expr) => {
        Parameter::Fixed($name.to_string(), $value)
    };
    ($name:expr) => {
        Parameter::Free($name.to_string())
    };
}

pub struct EML {
    data: Dataset,
    montecarlo: Dataset,
    amplitude_data: Box<dyn Node>,
    amplitude_montecarlo: Box<dyn Node>,
    parameters: Vec<Parameter>,
}

#[derive(Clone)]
pub enum Parameter {
    Free(String),
    Fixed(String, f64),
}

impl EML {
    #[anyinput]
    pub fn new(
        data: Dataset,
        montecarlo: Dataset,
        amplitude_data: Box<dyn Node>,
        amplitude_montecarlo: Box<dyn Node>,
        parameters: Vec<Parameter>,
    ) -> Self {
        let mut eml_instance = EML {
            data,
            montecarlo,
            amplitude_data,
            amplitude_montecarlo,
            parameters,
        };
        eml_instance.amplitude_data.resolve(&mut eml_instance.data);
        eml_instance
            .amplitude_montecarlo
            .resolve(&mut eml_instance.montecarlo);
        eml_instance
    }

    fn get_values(&self, vals: &Vec<f64>) -> HashMap<String, f64> {
        let mut result = HashMap::default();
        let mut val_iter = vals.iter().cloned().collect::<VecDeque<f64>>();
        for param in &self.parameters {
            match param {
                Parameter::Fixed(param_name, value) => {
                    result.insert(param_name.clone(), *value);
                }
                Parameter::Free(param_name) => {
                    if let Some(next_val) = val_iter.pop_front() {
                        result.insert(param_name.clone(), next_val);
                    } else {
                        panic!("Not enough free values provided!")
                    }
                }
            }
        }
        result
    }
}

impl CostFunction for EML {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let pars = self.get_values(param);
        let fn_data: f64 = self
            .amplitude_data
            .eval(&self.data, &pars)
            .into_par_iter()
            .zip(self.data.weights())
            .map(|(val, w)| val.re.ln() * w)
            .sum();
        let fn_montecarlo: f64 = self
            .amplitude_montecarlo
            .eval(&self.montecarlo, &pars)
            .into_par_iter()
            .zip(self.data.weights())
            .map(|(val, w)| val.re * w)
            .sum();
        #[allow(clippy::cast_precision_loss)]
        // Ok(-2.0 * (fn_data - ((self.data.len() / self.montecarlo.len()) as f64) * fn_montecarlo))
        Ok(-2.0
            * (fn_data - self.data.weighted_len() / self.montecarlo.weighted_len() * fn_montecarlo))
    }
}
