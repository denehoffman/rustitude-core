use anyinput::anyinput;
use num_complex::Complex64;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;

use crate::prelude::{Amplitude, AmplitudeContainer, Dataset};

#[derive(Default, Debug, Clone)]
pub struct ParameterManager {
    pub parameters: Vec<Vec<String>>,
    pub fixed: HashMap<String, f64>,
}

impl ParameterManager {
    pub fn free_parameters(&self) -> Vec<Vec<String>> {
        let mut output: Vec<Vec<String>> = Vec::new();
        for parameter_group in &self.parameters {
            let mut is_fixed = false;
            for parameter in parameter_group {
                if self.fixed.contains_key(parameter) {
                    is_fixed = true
                }
            }
            if !is_fixed {
                output.push(parameter_group.to_vec());
            }
        }
        output
    }

    fn to_vec(&self, values: Vec<f64>) -> HashMap<String, f64> {
        assert_eq!(
            self.free_parameters().len(),
            values.len(),
            "Number of input parameters ({}) does not match number of free parameters ({})",
            values.len(),
            self.free_parameters().len()
        );
        let mut output: HashMap<String, f64> = HashMap::default();
        let mut value_index = 0;

        for group in &self.parameters {
            let any_fixed = group.iter().any(|param| self.fixed.contains_key(param));
            for param in group {
                let entry = if any_fixed {
                    self.fixed.get(param).cloned()
                } else {
                    values.get(value_index).cloned()
                };
                if let Some(value) = entry {
                    output.insert(param.clone(), value);
                }
            }
            if !any_fixed {
                value_index += 1;
            }
        }
        output
    }

    #[anyinput]
    pub fn register(&mut self, name: AnyString) {
        for par_group in &self.parameters {
            if par_group.contains(&name.to_string()) {
                return;
            }
        }
        self.parameters.push(vec![name.to_string()]);
    }

    #[anyinput]
    pub fn constrain(&mut self, name_a: AnyString, name_b: AnyString) {
        let mut ind_a = None;
        let mut ind_b = None;
        for (index, par_group) in self.parameters.iter().enumerate() {
            if par_group.contains(&name_a.to_string()) {
                ind_a = Some(index);
            }
            if par_group.contains(&name_b.to_string()) {
                ind_b = Some(index);
            }
        }
        if let (Some(i_a), Some(i_b)) = (ind_a, ind_b) {
            if i_a == i_b {
                return;
            }

            let fixed_in_a = self.parameters[i_a]
                .iter()
                .any(|par_name| self.fixed.contains_key(par_name));
            let fixed_in_b = self.parameters[i_b]
                .iter()
                .any(|par_name| self.fixed.contains_key(par_name));

            if fixed_in_a {
                let fixed_value_a = *self.fixed.get(&self.parameters[i_a][0]).unwrap();
                if fixed_in_b {
                    let fixed_value_b = *self.fixed.get(&self.parameters[i_b][0]).unwrap();
                    if fixed_value_a != fixed_value_b {
                        panic!("Values differ");
                    }
                } else {
                    // Collect names to fix
                    let names_to_fix: Vec<String> = self.parameters[i_b].to_vec();

                    // Fix each name
                    for name in names_to_fix {
                        self.fix(name.clone(), fixed_value_a);
                    }
                }
            }

            // Move parameters from i_b to i_a
            let pars_b = self.parameters.remove(i_b);
            if let Some(pars_a) = self.parameters.get_mut(i_a) {
                pars_a.extend(pars_b);
            }
        }
    }

    #[anyinput]
    pub fn fix(&mut self, name: AnyString, value: f64) {
        let mut index = None;
        for (i, par_group) in self.parameters.iter().enumerate() {
            if par_group.contains(&name.to_string()) {
                index = Some(i);
            }
        }
        if let Some(i) = index {
            for par_name in &self.parameters[i] {
                self.fixed.entry(par_name.to_string()).or_insert(value);
            }
        }
    }

    #[anyinput]
    pub fn free(&mut self, name: &str) {
        self.fixed.remove(name);
    }
}

#[derive(Debug)]
pub struct Manager {
    pub parameters: ParameterManager,
    amplitudes: HashMap<String, Vec<AmplitudeContainer>>,
    sums: Vec<Vec<String>>,
    separator: String,
}

impl std::fmt::Display for Manager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Manager")?;
        writeln!(f, "parameters: {:?}", self.parameters)?;
        writeln!(f, "amplitudes:")?;
        for (key, amps) in self.amplitudes.iter() {
            writeln!(f, "\t{key}: ")?;
            for amp in amps {
                writeln!(f, "{amp}")?;
            }
        }
        writeln!(f, "sums: {:?}", self.sums)?;
        Ok(())
    }
}

impl Default for Manager {
    fn default() -> Self {
        Manager {
            parameters: ParameterManager::default(),
            amplitudes: HashMap::default(),
            sums: Vec::default(),
            separator: "::".to_string(),
        }
    }
}

impl Manager {
    #[anyinput]
    pub fn new(separator: AnyString) -> Self {
        Manager {
            separator: separator.to_string(),
            ..Default::default()
        }
    }

    #[anyinput]
    pub fn register<T>(&mut self, group_name: AnyString, amp_name: AnyString, amplitude: T)
    where
        T: Amplitude + 'static,
    {
        let amp_container = AmplitudeContainer::new(amp_name, amplitude);
        if let Some(par_names) = amp_container.parameter_names(&self.separator) {
            for par_name in par_names {
                self.parameters.register(par_name);
            }
        }
        self.parameters.register(format!("{} re", group_name));
        self.parameters.register(format!("{} im", group_name));
        self.parameters.register(format!("{} scale", group_name));
        self.parameters.fix(format!("{} scale", group_name), 1.0);
        self.amplitudes
            .entry(group_name.to_string())
            .or_default()
            .push(amp_container);
    }

    #[anyinput]
    pub fn constrain(&mut self, name_a: AnyString, name_b: AnyString) {
        self.parameters
            .constrain(format!("{} re", name_a), format!("{} re", name_b));
        self.parameters
            .constrain(format!("{} im", name_a), format!("{} im", name_b));
    }

    #[anyinput]
    pub fn fix_imag(&mut self, name: AnyString) {
        self.parameters.fix(format!("{} im", name), 0.0);
    }

    #[anyinput]
    pub fn anchor(&mut self, name: AnyString) {
        self.fix_imag(name)
    }

    pub fn sum(&mut self, summands: Vec<&'static str>) {
        for summand in &summands {
            if !self.amplitudes.contains_key(*summand) {
                panic!("The amplitude group '{}' has not been registered", summand);
            }
        }
        self.sums
            .push(summands.into_iter().map(|s| s.to_string()).collect());
    }

    pub fn evaluate(&mut self, values: Vec<f64>, dataset: &Dataset) -> Vec<f64> {
        let parameters = self.parameters.to_vec(values);
        let mut tot: Vec<f64> = vec![0.0; dataset.len()];
        for sum in &self.sums {
            let mut sum_tot: Vec<Complex64> = vec![0.0.into(); dataset.len()];
            for summand in sum {
                if let Some(amp_group) = self.amplitudes.get_mut(summand) {
                    let mut amp_group_tot: Vec<Complex64> = vec![1.0.into(); dataset.len()];
                    for amp in amp_group.iter_mut() {
                        let res = amp.evaluate(dataset, &parameters, &self.separator);
                        amp_group_tot
                            .par_iter_mut()
                            .zip(res)
                            .for_each(|(z, r)| *z *= r);
                    }
                    let p_re = format!("{} re", summand);
                    let p_im = format!("{} im", summand);
                    let p_scale = format!("{} scale", summand);
                    let amp_group_amplitude = Complex64::new(
                        *parameters.get(&p_re).unwrap_or_else(|| {
                            panic!(
                                "{} was not found in the list of registered parameters",
                                p_re
                            )
                        }),
                        *parameters.get(&p_im).unwrap_or_else(|| {
                            panic!(
                                "{} was not found in the list of registered parameters",
                                p_im
                            )
                        }),
                    );
                    let amp_group_scale = *parameters.get(&p_scale).unwrap_or_else(|| {
                        panic!(
                            "{} was not found in the list of registered parameters",
                            p_scale
                        )
                    });
                    sum_tot
                        .par_iter_mut()
                        .zip(amp_group_tot)
                        .for_each(|(z, r)| *z += r * amp_group_amplitude * amp_group_scale);
                } else {
                    panic!("A summand was not mapped to any registered amplitude group, use the 'Manager::sum' method to prevent this in the future");
                }
            }
            tot.par_iter_mut()
                .zip(sum_tot)
                .for_each(|(v, z)| *v += z.norm_sqr())
        }
        tot
    }
}
