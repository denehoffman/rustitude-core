use std::collections::VecDeque;

use rustc_hash::FxHashMap as HashMap;

use anyinput::anyinput;
use argmin::core::{CostFunction, Error};
use rayon::prelude::*;

use crate::{amplitude::Parameter, dataset::Dataset, node::Node};

#[derive(Clone)]
pub struct EML<'a, 'b, N>
where
    N: Node + Clone,
{
    pub data: &'a Dataset,
    pub montecarlo: &'b Dataset,
    pub amplitude: N,
    pub parameters: Vec<Parameter>,
}

impl<'a, 'b, N> EML<'a, 'b, N>
where
    N: Node + Clone,
{
    #[anyinput]
    pub fn new(
        data: &'a Dataset,
        montecarlo: &'b Dataset,
        amplitude: N,
        parameters: Vec<Parameter>,
    ) -> Self {
        EML {
            data,
            montecarlo,
            amplitude: amplitude.clone(),
            parameters,
        }
    }

    pub fn get_params(&self, vals: &[f64]) -> HashMap<String, f64> {
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

impl<'a, 'b, N> CostFunction for EML<'a, 'b, N>
where
    N: Node + Clone,
{
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let pars = self.get_params(param);
        let fn_data: f64 = self
            .amplitude
            .eval(self.data, &pars)
            .into_par_iter()
            .zip(self.data.weights())
            .map(|(val, w)| val.re.ln() * w)
            .sum();
        let fn_montecarlo: f64 = self
            .amplitude
            .eval(self.montecarlo, &pars)
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
