use std::collections::HashMap;

use anyinput::anyinput;
use argmin::core::{CostFunction, Error};
use rayon::prelude::*;

use crate::{dataset::Dataset, node::Node};

pub struct EML<'a, 'b, T, U>
where
    T: Node,
    U: Node,
{
    data: &'a Dataset,
    montecarlo: &'b Dataset,
    amplitude_data: T,
    amplitude_montecarlo: U,
    parameter_order: Vec<String>,
}

impl<'a, 'b, T, U> EML<'a, 'b, T, U>
where
    T: Node + Clone,
    U: Node + Clone,
{
    #[anyinput]
    pub fn new(
        data: &'a mut Dataset,
        montecarlo: &'b mut Dataset,
        amplitude_data: &T,
        amplitude_montecarlo: &U,
        parameters: Vec<&str>,
    ) -> Self {
        amplitude_data.resolve(data);
        amplitude_montecarlo.resolve(montecarlo);
        let parameter_order = parameters.into_iter().map(|s| s.to_string()).collect();
        EML {
            data,
            montecarlo,
            amplitude_data: amplitude_data.clone(),
            amplitude_montecarlo: amplitude_montecarlo.clone(),
            parameter_order,
        }
    }
}

impl<'a, 'b, T, U> CostFunction for EML<'a, 'b, T, U>
where
    T: Node,
    U: Node,
{
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let pars: HashMap<String, f64> = self
            .parameter_order
            .iter()
            .zip(param.into_iter())
            .map(|(name, val)| (name.clone(), *val))
            .collect();
        let fn_data: f64 = self
            .amplitude_data
            .eval(self.data, &pars)
            .into_par_iter()
            .zip(self.data.weights())
            .map(|(val, w)| val.re.ln() * w)
            .sum();
        let fn_montecarlo: f64 = self
            .amplitude_montecarlo
            .eval(self.montecarlo, &pars)
            .into_par_iter()
            .zip(self.data.weights())
            .map(|(val, w)| val.re.powf(*w))
            .sum();
        #[allow(clippy::cast_precision_loss)]
        Ok(-2.0 * (fn_data - (self.data.len() / self.montecarlo.len()) as f64 * fn_montecarlo))
    }
}
