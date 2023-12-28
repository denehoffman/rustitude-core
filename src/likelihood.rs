use argmin::core::{CostFunction, Error};
use rayon::prelude::*;

use crate::prelude::{Amplitude, Dataset, ParameterType};

pub struct ParallelExtendedMaximumLikelihood<'a> {
    data: Dataset,
    montecarlo: Dataset,
    amplitude: Amplitude,
    parameters: Vec<ParameterType<'a>>,
}

impl<'a> ParallelExtendedMaximumLikelihood<'a> {
    fn setup(&mut self) {
        self.amplitude.par_resolve_dependencies(&mut self.data);
        self.amplitude
            .par_resolve_dependencies(&mut self.montecarlo);
    }
}

fn params_from_vec<'a>(vals: &[f64], pars: &Vec<ParameterType<'a>>) -> Vec<ParameterType<'a>> {
    let mut i: usize = 0;
    let mut new_pars: Vec<ParameterType> = Vec::new();
    for partype in pars {
        match partype {
            ParameterType::Scalar(par) => {
                let mut new_par = *par;
                new_par.value = vals[i];
                new_pars.push(new_par.into());
                i += 1;
            }
            ParameterType::CScalar(par) => {
                let mut new_par = *par;
                new_par.a = vals[i];
                new_par.b = vals[i + 1];
                new_pars.push(new_par.into());
                i += 2;
            }
        }
    }
    new_pars
}

impl<'a> CostFunction for ParallelExtendedMaximumLikelihood<'a> {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, params: &Self::Param) -> Result<Self::Output, Error> {
        let pars = params_from_vec(params, &self.parameters);
        let fn_data: f64 = self
            .amplitude
            .par_evaluate_on(&pars, &self.data)
            .into_par_iter()
            .map(|val| val.re.ln())
            .sum();
        let fn_mc: f64 = self
            .amplitude
            .par_evaluate_on(&pars, &self.montecarlo)
            .into_par_iter()
            .map(|val| val.re)
            .sum();
        Ok(-2.0
            * (fn_data - (self.data.n_entries as f64 / self.montecarlo.n_entries as f64) * fn_mc))
    }
}
