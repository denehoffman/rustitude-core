use crate::{dataset::Dataset, prelude::Event};
use num_complex::Complex64;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use uuid::Uuid;

// Problem: Parameter names are only unique within the Amplitude
// Solution: We pair them with a UUID from the Amplitude
//
// We need to register amplitudes into groups. These groups will need to have a name so we can add
// more amplitudes to them. Amplitudes in a group are multiplied, groups in a sum are added. Each
// group has an associated Re, Im, and Scale parameter. Maybe group should be a struct?
//
// If group is a struct, what does it contain? It could hold the name. Will we be able to hold
// multiple Amplitudes with different lifetimes?

#[derive(Hash, PartialEq, Eq)]
pub struct Parameter(Uuid, String);
impl Parameter {
    pub fn new(uuid: Uuid, name: &str) -> Self {
        Self(uuid, name.to_string())
    }
}

pub struct ParameterManager {
    parameter_store: HashMap<Uuid, Vec<String>>,
    amplitudes: Vec<Uuid>,
    groupings: Vec<Vec<(Uuid, String)>>,
}

impl ParameterManager {
    pub fn register<'d, 'n, N, E>(&mut self, name: &str, amplitude: Amplitude<'d, 'n, N, E>)
    where
        N: Node<E>,
        E: Event,
    {
        let uuid = amplitude.uuid;
        let parameters = amplitude.node.parameters();
    }
    pub fn get_parameters(&self, inputs: Vec<f64>) -> HashMap<Uuid, HashMap<String, f64>> {
        HashMap::default()
    }
}

pub struct Amplitude<'d, 'n, N, E>
where
    N: Node<E>,
    E: Event,
{
    dataset: &'d Dataset<E>,
    node: &'n mut N,
    pub uuid: Uuid,
}

impl<'d, 'n, N, E> Amplitude<'d, 'n, N, E>
where
    N: Node<E> + Sync,
    E: Event,
{
    pub fn new(dataset: &'d Dataset<E>, node: &'n mut N) -> Self {
        Self {
            dataset,
            node,
            uuid: Uuid::new_v4(),
        }
    }
    pub fn precompute(&mut self) {
        self.node.precalculate(&self.dataset);
    }
    pub fn compute(&self, parameters: &HashMap<Uuid, HashMap<String, f64>>) -> Vec<Complex64> {
        let parameter_map = &parameters[&self.uuid];
        if let Some(parameter_order) = self.node.parameters() {
            let parameter_vec = parameter_order.iter().map(|k| parameter_map[k]).collect();
            self.dataset
                .par_iter()
                .enumerate()
                .map(|(i, event)| self.node.calculate(i, event, &parameter_vec))
                .collect()
        } else {
            let parameter_vec = Vec::default();
            self.dataset
                .par_iter()
                .enumerate()
                .map(|(i, event)| self.node.calculate(i, event, &parameter_vec))
                .collect()
        }
    }
}

pub trait Node<E: Event> {
    fn precalculate(&mut self, dataset: &Dataset<E>);
    fn calculate(&self, index: usize, event: &E, parameters: &Vec<f64>) -> Complex64;
    fn parameters(&self) -> Option<Vec<String>>;
}
