use rayon::prelude::*;
// use std::collections::HashMap;
use rustc_hash::FxHashMap as HashMap;

use ndarray::{Array1, Array2};
use num_complex::Complex64;

use crate::{amplitude::Variable, four_momentum::FourMomentum, prelude::VarMap};
use derive_more::IsVariant;

#[derive(Debug, Clone, IsVariant)]
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
    /// Get the internal structure of a field.
    ///
    /// This function takes an enum `FieldType` and extracts a reference to the underlying
    /// `FieldType::Index` if it is an instance of that variant or panics otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the underlying variant is not a `ParameterType::Index`.
    pub fn index(&self) -> &usize {
        if let Self::Index(value) = self {
            value
        } else {
            panic!("Could not convert to Index type")
        }
    }

    /// Get the internal structure of a field.
    ///
    /// This function takes an enum `FieldType` and extracts a reference to the underlying
    /// `FieldType::Scalar` if it is an instance of that variant or panics otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the underlying variant is not a `ParameterType::Scalar`.
    pub fn scalar(&self) -> &f64 {
        if let Self::Scalar(value) = self {
            value
        } else {
            panic!("Could not convert to Scalar type")
        }
    }

    /// Get the internal structure of a field.
    ///
    /// This function takes an enum `FieldType` and extracts a reference to the underlying
    /// `FieldType::CScalar` if it is an instance of that variant or panics otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the underlying variant is not a `ParameterType::CScalar`.
    pub fn cscalar(&self) -> &Complex64 {
        if let Self::CScalar(value) = self {
            value
        } else {
            panic!("Could not convert to CScalar type")
        }
    }

    /// Get the internal structure of a field.
    ///
    /// This function takes an enum `FieldType` and extracts a reference to the underlying
    /// `FieldType::Vector` if it is an instance of that variant or panics otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the underlying variant is not a `ParameterType::Vector`.
    pub fn vector(&self) -> &Array1<f64> {
        if let Self::Vector(value) = self {
            value
        } else {
            panic!("Could not convert to Vector type")
        }
    }

    /// Get the internal structure of a field.
    ///
    /// This function takes an enum `FieldType` and extracts a reference to the underlying
    /// `FieldType::CVector` if it is an instance of that variant or panics otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the underlying variant is not a `ParameterType::CVector`.
    pub fn cvector(&self) -> &Array1<Complex64> {
        if let Self::CVector(value) = self {
            value
        } else {
            panic!("Could not convert to CVector type")
        }
    }
    /// Get the internal structure of a field.
    ///
    /// This function takes an enum `FieldType` and extracts a reference to the underlying
    /// `FieldType::Matrix` if it is an instance of that variant or panics otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the underlying variant is not a `ParameterType::Matrix`.
    pub fn matrix(&self) -> &Array2<f64> {
        if let Self::Matrix(value) = self {
            value
        } else {
            panic!("Could not convert to Matrix type")
        }
    }

    /// Get the internal structure of a field.
    ///
    /// This function takes an enum `FieldType` and extracts a reference to the underlying
    /// `FieldType::CMatrix` if it is an instance of that variant or panics otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the underlying variant is not a `ParameterType::CMatrix`.
    pub fn cmatrix(&self) -> &Array2<Complex64> {
        if let Self::CMatrix(value) = self {
            value
        } else {
            panic!("Could not convert to CMatrix type")
        }
    }

    /// Get the internal structure of a field.
    ///
    /// This function takes an enum `FieldType` and extracts a reference to the underlying
    /// `FieldType::Momentum` if it is an instance of that variant or panics otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the underlying variant is not a `ParameterType::Momentum`.
    pub fn momentum(&self) -> &FourMomentum {
        if let Self::Momentum(value) = self {
            value
        } else {
            panic!("Could not convert to Momentum type")
        }
    }

    /// Get the internal structure of a field.
    ///
    /// This function takes an enum `FieldType` and extracts a reference to the underlying
    /// `FieldType::MomentumVec` if it is an instance of that variant or panics otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the underlying variant is not a `ParameterType::MomentumVec`.
    pub fn momenta(&self) -> &[FourMomentum] {
        if let Self::MomentumVec(value) = self {
            value
        } else {
            panic!("Could not convert to MomentumVec type")
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
    pub fn add_field(&mut self, name: &str, field: &[FieldType], prunable: bool) {
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
            self.add_field(&variable.name, &field, prunable);
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
            self.add_field(&variable.name, &field, prunable);
        }
    }
}
