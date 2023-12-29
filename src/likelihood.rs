use argmin::core::{CostFunction, Error};
use rayon::prelude::*;

use crate::prelude::{Amplitude, Dataset, Parameter};

pub struct ParallelExtendedMaximumLikelihood<'a> {
    pub data: Dataset,
    pub amplitude_data: Amplitude<'a>,
    pub montecarlo: Dataset,
    pub amplitude_montecarlo: Amplitude<'a>,
    pub parameter_order: Vec<Parameter<'a>>,
}

impl<'a> ParallelExtendedMaximumLikelihood<'a> {
    pub fn setup(&mut self) {
        self.amplitude_data.par_resolve_dependencies(&mut self.data);
        self.amplitude_montecarlo
            .par_resolve_dependencies(&mut self.montecarlo);
    }
}

impl<'a> CostFunction for ParallelExtendedMaximumLikelihood<'a> {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, params: &Self::Param) -> Result<Self::Output, Error> {
        let par_names: Vec<&str> = self.parameter_order.iter().map(|p| p.name).collect();
        self.amplitude_data.load_params(params, &par_names);
        self.amplitude_montecarlo.load_params(params, &par_names);
        let fn_data: f64 = self
            .amplitude_data
            .par_evaluate_on(&self.data)
            .into_par_iter()
            .map(|val| val.re.ln())
            .sum();
        let fn_mc: f64 = self
            .amplitude_montecarlo
            .par_evaluate_on(&self.montecarlo)
            .into_par_iter()
            .map(|val| val.re)
            .sum();
        #[allow(clippy::cast_precision_loss)]
        Ok(-2.0
            * (fn_data - (self.data.n_entries as f64 / self.montecarlo.n_entries as f64) * fn_mc))
    }
}
