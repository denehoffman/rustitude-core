use rayon::prelude::*;
use std::collections::HashMap;

use ndarray::{Array1, Array2};
use num_complex::Complex64;

use crate::{amplitude::Variable, four_momentum::FourMomentum, prelude::VarMap};

#[derive(Debug, Clone)]
pub enum FieldType {
    Index(usize),
    Scalar(f64),
    CScalar(Complex64),
    Vector(Array1<f64>),
    CVector(Array1<Complex64>),
    Matrix(Array2<f64>),
    CMatrix(Array2<Complex64>),
    Momentum(FourMomentum),
    MomentumVec(Vec<FourMomentum>),
}

impl FieldType {
    pub fn index(&self) -> Result<&usize, &str> {
        if let Self::Index(value) = self {
            Ok(value)
        } else {
            Err("Could not convert to Index type")
        }
    }
    pub fn scalar(&self) -> Result<&f64, &str> {
        if let Self::Scalar(value) = self {
            Ok(value)
        } else {
            Err("Could not convert to Scalar type")
        }
    }
    pub fn cscalar(&self) -> Result<&Complex64, &str> {
        if let Self::CScalar(value) = self {
            Ok(value)
        } else {
            Err("Could not convert to CScalar type")
        }
    }
    pub fn vector(&self) -> Result<&Array1<f64>, &str> {
        if let Self::Vector(value) = self {
            Ok(value)
        } else {
            Err("Could not convert to Vector type")
        }
    }
    pub fn cvector(&self) -> Result<&Array1<Complex64>, &str> {
        if let Self::CVector(value) = self {
            Ok(value)
        } else {
            Err("Could not convert to CVector type")
        }
    }
    pub fn matrix(&self) -> Result<&Array2<f64>, &str> {
        if let Self::Matrix(value) = self {
            Ok(value)
        } else {
            Err("Could not convert to Matrix type")
        }
    }
    pub fn cmatrix(&self) -> Result<&Array2<Complex64>, &str> {
        if let Self::CMatrix(value) = self {
            Ok(value)
        } else {
            Err("Could not convert to CMatrix type")
        }
    }
    pub fn momentum(&self) -> Result<&FourMomentum, &str> {
        if let Self::Momentum(value) = self {
            Ok(value)
        } else {
            Err("Could not convert to Momentum type")
        }
    }
    pub fn momenta(&self) -> Result<&[FourMomentum], &str> {
        if let Self::MomentumVec(value) = self {
            Ok(value)
        } else {
            Err("Could not convert to MomentumVec type")
        }
    }
}

#[derive(Debug)]
pub struct Dataset {
    pub entries: Vec<VarMap>,
    pub n_entries: usize,
    prunable: HashMap<String, bool>,
}

impl Dataset {
    pub fn new(n_entries: usize) -> Self {
        let entries: Vec<VarMap> = (0..n_entries)
            .map(|i| {
                let mut entry = HashMap::default();
                entry.insert("Index".to_string(), FieldType::Index(i));
                entry
            })
            .collect();
        let mut prunable = HashMap::default();
        prunable.insert("Index".to_string(), false);
        Self {
            entries,
            n_entries,
            prunable,
        }
    }
    pub fn add_field(&mut self, name: &str, field: Vec<FieldType>, prunable: bool) {
        for (i, entry) in &mut self.entries.iter_mut().enumerate() {
            entry.insert(name.to_string(), field[i].clone());
        }
        self.prunable.insert(name.to_string(), prunable);
    }

    pub fn remove_field(&mut self, name: &str) {
        for entry in &mut self.entries {
            entry.remove(name);
        }
    }

    pub fn prune(&mut self) {
        let prunable_entries: Vec<String> = self
            .prunable
            .iter()
            .filter(|&(_, &can_prune)| can_prune)
            .map(|(key, _)| key.clone())
            .collect();
        for prunable_entry in &prunable_entries {
            self.remove_field(prunable_entry);
        }
    }

    pub fn resolve_dependencies(&mut self, variable: Variable, prunable: bool) {
        // first resolve any subdependencies the variable has (and recurse)
        if let Some(deps) = variable.dependencies {
            for dep in deps {
                if !self.prunable.contains_key(&*dep.name) {
                    self.resolve_dependencies(dep, true);
                }
            }
        }
        // then resolve the variable itself
        let fn_lock = variable.function.read();
        if !self.prunable.contains_key(&*variable.name) {
            #[allow(clippy::redundant_closure)]
            let field: Vec<FieldType> = self.entries.iter().map(|entry| fn_lock(entry)).collect();
            self.add_field(&variable.name, field, prunable);
        }
    }
    pub fn par_resolve_dependencies(&mut self, variable: Variable, prunable: bool) {
        // first resolve any subdependencies the variable has (and recurse)
        if let Some(deps) = variable.dependencies {
            for dep in deps {
                if !self.prunable.contains_key(&*dep.name) {
                    self.par_resolve_dependencies(dep, true);
                }
            }
        }
        // then resolve the variable itself
        let fn_lock = variable.function.read();
        if !self.prunable.contains_key(&*variable.name) {
            #[allow(clippy::redundant_closure)]
            let field: Vec<FieldType> = self
                .entries
                .par_iter()
                .map(|entry| fn_lock(entry))
                .collect();
            self.add_field(&variable.name, field, prunable);
        }
    }
}
