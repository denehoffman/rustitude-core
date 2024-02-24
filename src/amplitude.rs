use crate::{
    dataset::{CScalar64, Dataset},
    node::{
        ComplexParameterNode, ConstantNode, Dependent, MulNode, Node, Parameterized, Resolvable,
    },
};
use anyinput::anyinput;
use derive_builder::Builder;
use num_complex::Complex64;
use num_traits::Zero;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;

#[derive(Builder)]
pub struct Amplitude<N>
where
    N: Node,
{
    #[builder(setter(into))]
    name: String,
    expr: N,
    #[builder(default = "self.default_param()")]
    param: ComplexParameterNode,
    #[builder(default = "self.default_node()")]
    node: MulNode<ComplexParameterNode, N>,
}

impl<N> AmplitudeBuilder<N>
where
    N: Node + Clone,
{
    fn default_param(&self) -> ComplexParameterNode {
        ComplexParameterNode::new(
            format!("{} re", self.name.clone().unwrap()),
            format!("{} im", self.name.clone().unwrap()),
        )
    }

    fn default_node(&self) -> MulNode<ComplexParameterNode, N> {
        self.param.clone().unwrap().mul(&self.expr.clone().unwrap())
    }
}

pub struct AmplitudeSum<N>
where
    N: Node,
{
    pub amplitudes: Vec<Amplitude<N>>,
}

impl<N> AmplitudeSum<N>
where
    N: Node + Clone,
{
    pub fn push(&mut self, amplitude: Amplitude<N>) {
        self.amplitudes.push(amplitude);
    }

    pub fn resolve(&self, dataset: &mut Dataset) {
        for amp in &self.amplitudes {
            amp.node.resolve(dataset);
        }
    }

    pub fn eval(&self, names: &Vec<String>, ds: &Dataset, pars: &HashMap<String, f64>) -> Vec<f64> {
        let mut res = vec![Complex64::zero(); ds.len()];
        for amp in &self.amplitudes {
            if names.contains(&amp.name) {
                res = amp
                    .node
                    .eval(ds, pars)
                    .par_iter()
                    .zip(res)
                    .map(|(v, res)| v + res)
                    .collect()
            }
        }
        res.iter().map(|v| v.norm_sqr()).collect()
    }
}

#[derive(Default)]
pub struct AmplitudeManager<N>
where
    N: Node,
{
    pub sums: Vec<AmplitudeSum<N>>,
}

impl<N> AmplitudeManager<N>
where
    N: Node + Clone,
{
    pub fn resolve(&self, ds: &mut Dataset) {
        for sum in &self.sums {
            sum.resolve(ds);
        }
    }

    pub fn eval(&self, names: &Vec<String>, ds: &Dataset, pars: &HashMap<String, f64>) -> Vec<f64> {
        let mut res = vec![Complex64::zero(); ds.len()];
        for sum in &self.sums {
            res = sum
                .eval(names, ds, pars)
                .iter()
                .zip(res)
                .map(|(v, res)| v + res)
                .collect();
        }
        res.par_iter().map(|v| v.re).collect()
    }
}

#[macro_export]
macro_rules! cohsum {
    ($($amp:expr),*) => {
        AmplitudeSum { amplitudes: vec![$($amp),*] }
    };
}

#[macro_export]
macro_rules! manager {
    ($($sum:expr),*) => {
        AmplitudeManager { sums: vec![$($sum),*]}
    }
}

#[macro_export]
macro_rules! par {
    ($name:expr, $value:expr) => {
        Parameter::Fixed($name.to_string(), $value)
    };
    ($name:expr) => {
        Parameter::Free($name.to_string())
    };
}

#[derive(Clone)]
pub enum Parameter {
    Free(String),
    Fixed(String, f64),
}
