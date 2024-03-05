use std::fmt::Display;

use anyinput::anyinput;
use num_complex::Complex64;
use rustc_hash::FxHashMap as HashMap;

use crate::prelude::Dataset;

pub trait Amplitude: Display {
    fn evaluate(&mut self, dataset: &Dataset, parameters: &HashMap<String, f64>) -> Vec<Complex64>;
    fn parameter_names(&self) -> Option<Vec<String>> {
        None
    }
}

pub struct AmplitudeContainer {
    pub name: String,
    pub amplitude: Box<dyn Amplitude>,
}

impl std::fmt::Debug for AmplitudeContainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Amplitude: {}", self.name)
    }
}

impl std::fmt::Display for AmplitudeContainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Amplitude: {} -> {}", self.name, self.amplitude)
    }
}

impl AmplitudeContainer {
    #[anyinput]
    pub fn new<T>(name: AnyString, amplitude: T) -> Self
    where
        T: Amplitude + 'static,
    {
        AmplitudeContainer {
            name: name.to_string(),
            amplitude: Box::new(amplitude),
        }
    }

    pub fn parameter_names(&self, separator: &String) -> Option<Vec<String>> {
        self.amplitude.parameter_names().map(|par_names| {
            par_names
                .iter()
                .map(|par_name| format!("{}{}{}", self.name, separator, par_name))
                .collect()
        })
    }

    pub fn evaluate(
        &mut self,
        dataset: &Dataset,
        parameters: &HashMap<String, f64>,
        separator: &String,
    ) -> Vec<Complex64> {
        let mut amp_parameters: HashMap<String, f64> = HashMap::default();

        for (key, value) in parameters.iter() {
            if let Some(index) = key.find(separator) {
                let (prefix, _) = key.split_at(index);
                if prefix == self.name {
                    // Get the part after the separator
                    let (_, suffix) = key.split_at(index + separator.len());
                    // Insert into the new HashMap
                    amp_parameters.insert(suffix.to_string(), *value);
                }
            }
        }
        self.amplitude.evaluate(dataset, &amp_parameters)
    }
}
