use std::fs::File;

use polars::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;

use ndarray::{Array1, Array2};
use num_complex::Complex64;

use crate::{amplitude::Variable, four_momentum::FourMomentum, prelude::VarMap};
use variantly::Variantly;

#[derive(Debug, Clone, Variantly)]
pub enum FieldType {
    Index(usize),
    Scalar(f64),
    #[variantly(rename = "cscalar")]
    CScalar(Complex64),
    Vector(Array1<f64>),
    #[variantly(rename = "cvector")]
    CVector(Array1<Complex64>),
    Matrix(Array2<f64>),
    #[variantly(rename = "cmatrix")]
    CMatrix(Array2<Complex64>),
    Momentum(FourMomentum),
    #[variantly(rename = "momenta")]
    MomentumVec(Vec<FourMomentum>),
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

pub fn open_parquet(path: &str) -> Result<DataFrame, PolarsError> {
    //! Opens a parquet file from a path string
    //!
    //! # Errors
    //! Returns an error if the result isn't readable as a parquet file or if the file is not found.
    let file = File::open(path)?;
    ParquetReader::new(file).finish()
}

#[derive(Clone, Copy)]
pub enum PolarsTypeConversion {
    F32ToScalar,
    F64ToScalar,
    ListToVector,
}

pub fn extract_field(
    column_name: &str,
    column_type: PolarsTypeConversion,
    df: &DataFrame,
) -> Result<Vec<FieldType>, PolarsError> {
    //! Converts a Polars [`polars::prelude::Series`] into a [`Vec<FieldType>`] according to
    //! a conversion rule.
    //!
    //! # Panics
    //!
    //! Currently will panic if the branch/column name was not found in the
    //! [`polars::prelude::DataFrame`]
    //!
    //! # Errors
    //!
    //! Returns [`PolarsError`] if any step in the conversion fails.
    let series = df
        .column(column_name)
        .unwrap_or_else(|_| panic!("No branch {column_name}"));
    match column_type {
        PolarsTypeConversion::F32ToScalar => Ok(series
            .f32()?
            .to_vec()
            .into_iter()
            .map(|x| FieldType::Scalar(x.unwrap().into()))
            .collect::<Vec<FieldType>>()),
        PolarsTypeConversion::F64ToScalar => Ok(series
            .f64()?
            .to_vec()
            .into_iter()
            .map(|x| FieldType::Scalar(x.unwrap()))
            .collect::<Vec<FieldType>>()),
        PolarsTypeConversion::ListToVector => Ok(series
            .list()?
            .into_iter()
            .map(|x| {
                FieldType::Vector(Array1::from_vec(
                    x.unwrap()
                        .f32()
                        .unwrap()
                        .to_vec()
                        .into_iter()
                        .map(|x| x.unwrap().into())
                        .collect(),
                ))
            })
            .collect::<Vec<FieldType>>()),
    }
}
