use std::fs::File;

use dashmap::DashMap;
use polars::prelude::*;
use rayon::prelude::*;

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rustc_hash::FxHashMap;

use crate::{
    four_momentum::FourMomentum,
};
use anyinput::anyinput;
use thiserror::Error;
use variantly::Variantly;

#[derive(Debug, Copy, Clone, Variantly)]
pub enum DataType {
    #[variantly(rename = "scalar")]
    Scalar,
    #[variantly(rename = "cscalar")]
    CScalar,
    #[variantly(rename = "vector")]
    Vector,
    #[variantly(rename = "cvector")]
    CVector,
    #[variantly(rename = "matrix")]
    Matrix,
    #[variantly(rename = "cmatrix")]
    CMatrix,
    #[variantly(rename = "momentum")]
    Momentum,
    #[variantly(rename = "momentum_vector")]
    MomentumVector,
}

pub type Scalar64 = f64;
pub type CScalar64 = Complex64;
pub type Vector64 = Array1<f64>;
pub type CVector64 = Array1<Complex64>;
pub type Matrix64 = Array2<f64>;
pub type CMatrix64 = Array2<Complex64>;
pub type Momentum64 = FourMomentum;
pub type Momenta64 = Vec<FourMomentum>;

#[derive(Default, Debug)]
pub struct Entry {
    index: usize,
    scalar_map: DashMap<String, Scalar64>,
    cscalar_map: DashMap<String, CScalar64>,
    vector_map: DashMap<String, Vector64>,
    cvector_map: DashMap<String, CVector64>,
    matrix_map: DashMap<String, Matrix64>,
    cmatrix_map: DashMap<String, CMatrix64>,
    momentum_map: DashMap<String, Momentum64>,
    momenta_map: DashMap<String, Momenta64>,
}

#[derive(Error, Debug)]
pub enum DatasetError {
    #[error("No {data_type:?} exists with name {key_name:?}")]
    TypeError {
        key_name: String,
        data_type: DataType,
    },
    #[error("{key_name:?} already exists: {key_name:?}@{data_type:?}")]
    FieldExistsError {
        key_name: String,
        data_type: DataType,
    },
    #[error("Cannot store a field of size {field_size} in a dataset of size {dataset_size}")]
    IncompatibleSizesError {
        dataset_size: usize,
        field_size: usize,
    },
}

impl Entry {
    pub fn new(index: usize) -> Entry {
        Entry {
            index,
            ..Default::default()
        }
    }
    #[anyinput]
    pub fn scalar(&self, key: AnyString) -> Result<Scalar64, DatasetError> {
        match self.scalar_map.get(key) {
            Some(val) => Ok(*val),
            None => Err(DatasetError::TypeError {
                key_name: key.to_string(),
                data_type: DataType::Scalar,
            }),
        }
    }

    #[anyinput]
    pub fn cscalar(&self, key: AnyString) -> Result<CScalar64, DatasetError> {
        match self.cscalar_map.get(key) {
            Some(val) => Ok(*val),
            None => Err(DatasetError::TypeError {
                key_name: key.to_string(),
                data_type: DataType::CScalar,
            }),
        }
    }

    #[anyinput]
    pub fn vector(&self, key: AnyString) -> Result<Vector64, DatasetError> {
        match self.vector_map.get(key) {
            Some(val) => Ok(val.clone()),
            None => Err(DatasetError::TypeError {
                key_name: key.to_string(),
                data_type: DataType::Vector,
            }),
        }
    }

    #[anyinput]
    pub fn cvector(&self, key: AnyString) -> Result<CVector64, DatasetError> {
        match self.cvector_map.get(key) {
            Some(val) => Ok(val.clone()),
            None => Err(DatasetError::TypeError {
                key_name: key.to_string(),
                data_type: DataType::CVector,
            }),
        }
    }

    #[anyinput]
    pub fn matrix(&self, key: AnyString) -> Result<Matrix64, DatasetError> {
        match self.matrix_map.get(key) {
            Some(val) => Ok(val.clone()),
            None => Err(DatasetError::TypeError {
                key_name: key.to_string(),
                data_type: DataType::Matrix,
            }),
        }
    }

    #[anyinput]
    pub fn cmatrix(&self, key: AnyString) -> Result<CMatrix64, DatasetError> {
        match self.cmatrix_map.get(key) {
            Some(val) => Ok(val.clone()),
            None => Err(DatasetError::TypeError {
                key_name: key.to_string(),
                data_type: DataType::CMatrix,
            }),
        }
    }

    #[anyinput]
    pub fn momentum(&self, key: AnyString) -> Result<Momentum64, DatasetError> {
        match self.momentum_map.get(key) {
            Some(val) => Ok(*val),
            None => Err(DatasetError::TypeError {
                key_name: key.to_string(),
                data_type: DataType::Momentum,
            }),
        }
    }

    #[anyinput]
    pub fn momenta(&self, key: AnyString) -> Result<Momenta64, DatasetError> {
        match self.momenta_map.get(key) {
            Some(val) => Ok(val.clone()),
            None => Err(DatasetError::TypeError {
                key_name: key.to_string(),
                data_type: DataType::MomentumVector,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FieldType {
    datatype: DataType,
    prunable: bool,
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub entries: Vec<Arc<Entry>>,
    fields: FxHashMap<String, FieldType>,
}

impl Dataset {
    pub fn from_size(size: usize) -> Dataset {
        let entries = (0..size)
            .into_iter()
            .map(|index| Arc::new(Entry::new(index)))
            .collect();
        Dataset {
            entries,
            fields: FxHashMap::default(),
        }
    }
    pub fn from_size_par(size: usize) -> Dataset {
        let entries = (0..size)
            .into_par_iter()
            .map(|index| Arc::new(Entry::new(index)))
            .collect();
        Dataset {
            entries,
            fields: FxHashMap::default(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[anyinput]
    pub fn contains_field(&self, name: AnyString) -> bool {
        self.fields.contains_key(name)
    }

    #[anyinput]
    pub fn add_scalar_field(
        &mut self,
        name: AnyString,
        values: Vec<Scalar64>,
        prunable: bool,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        match self.fields.get(name) {
            Some(ft) => Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
                data_type: ft.datatype,
            }),
            None => {
                self.entries
                    .iter_mut()
                    .zip(values.into_iter())
                    .for_each(|(entry, value)| {
                        entry.scalar_map.insert(name.to_string(), value);
                    });

                self.fields.insert(
                    name.to_string(),
                    FieldType {
                        datatype: DataType::Scalar,
                        prunable,
                    },
                );
                Ok(())
            }
        }
    }

    #[anyinput]
    pub fn add_scalar_field_par(
        &mut self,
        name: AnyString,
        values: Vec<Scalar64>,
        prunable: bool,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        match self.fields.get(name) {
            Some(ft) => Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
                data_type: ft.datatype,
            }),
            None => {
                self.entries
                    .par_iter_mut()
                    .zip(values.into_par_iter())
                    .for_each(|(entry, value)| {
                        entry.scalar_map.insert(name.to_string(), value);
                    });

                self.fields.insert(
                    name.to_string(),
                    FieldType {
                        datatype: DataType::Scalar,
                        prunable,
                    },
                );
                Ok(())
            }
        }
    }

    #[anyinput]
    pub fn add_cscalar_field(
        &mut self,
        name: AnyString,
        values: Vec<CScalar64>,
        prunable: bool,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        match self.fields.get(name) {
            Some(ft) => Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
                data_type: ft.datatype,
            }),
            None => {
                self.entries
                    .iter_mut()
                    .zip(values.into_iter())
                    .for_each(|(entry, value)| {
                        entry.cscalar_map.insert(name.to_string(), value);
                    });

                self.fields.insert(
                    name.to_string(),
                    FieldType {
                        datatype: DataType::CScalar,
                        prunable,
                    },
                );
                Ok(())
            }
        }
    }

    #[anyinput]
    pub fn add_cscalar_field_par(
        &mut self,
        name: AnyString,
        values: Vec<CScalar64>,
        prunable: bool,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        match self.fields.get(name) {
            Some(ft) => Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
                data_type: ft.datatype,
            }),
            None => {
                self.entries
                    .par_iter_mut()
                    .zip(values.into_par_iter())
                    .for_each(|(entry, value)| {
                        entry.cscalar_map.insert(name.to_string(), value);
                    });

                self.fields.insert(
                    name.to_string(),
                    FieldType {
                        datatype: DataType::CScalar,
                        prunable,
                    },
                );
                Ok(())
            }
        }
    }

    #[anyinput]
    pub fn add_vector_field(
        &mut self,
        name: AnyString,
        values: Vec<Vector64>,
        prunable: bool,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        match self.fields.get(name) {
            Some(ft) => Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
                data_type: ft.datatype,
            }),
            None => {
                self.entries
                    .iter_mut()
                    .zip(values.into_iter())
                    .for_each(|(entry, value)| {
                        entry.vector_map.insert(name.to_string(), value);
                    });

                self.fields.insert(
                    name.to_string(),
                    FieldType {
                        datatype: DataType::Vector,
                        prunable,
                    },
                );
                Ok(())
            }
        }
    }

    #[anyinput]
    pub fn add_vector_field_par(
        &mut self,
        name: AnyString,
        values: Vec<Vector64>,
        prunable: bool,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        match self.fields.get(name) {
            Some(ft) => Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
                data_type: ft.datatype,
            }),
            None => {
                self.entries
                    .par_iter_mut()
                    .zip(values.into_par_iter())
                    .for_each(|(entry, value)| {
                        entry.vector_map.insert(name.to_string(), value);
                    });

                self.fields.insert(
                    name.to_string(),
                    FieldType {
                        datatype: DataType::Vector,
                        prunable,
                    },
                );
                Ok(())
            }
        }
    }

    #[anyinput]
    pub fn add_cvector_field(
        &mut self,
        name: AnyString,
        values: Vec<CVector64>,
        prunable: bool,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        match self.fields.get(name) {
            Some(ft) => Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
                data_type: ft.datatype,
            }),
            None => {
                self.entries
                    .iter_mut()
                    .zip(values.into_iter())
                    .for_each(|(entry, value)| {
                        entry.cvector_map.insert(name.to_string(), value);
                    });

                self.fields.insert(
                    name.to_string(),
                    FieldType {
                        datatype: DataType::CVector,
                        prunable,
                    },
                );
                Ok(())
            }
        }
    }

    #[anyinput]
    pub fn add_cvector_field_par(
        &mut self,
        name: AnyString,
        values: Vec<CVector64>,
        prunable: bool,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        match self.fields.get(name) {
            Some(ft) => Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
                data_type: ft.datatype,
            }),
            None => {
                self.entries
                    .par_iter_mut()
                    .zip(values.into_par_iter())
                    .for_each(|(entry, value)| {
                        entry.cvector_map.insert(name.to_string(), value);
                    });

                self.fields.insert(
                    name.to_string(),
                    FieldType {
                        datatype: DataType::CVector,
                        prunable,
                    },
                );
                Ok(())
            }
        }
    }

    #[anyinput]
    pub fn add_matrix_field(
        &mut self,
        name: AnyString,
        values: Vec<Matrix64>,
        prunable: bool,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        match self.fields.get(name) {
            Some(ft) => Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
                data_type: ft.datatype,
            }),
            None => {
                self.entries
                    .iter_mut()
                    .zip(values.into_iter())
                    .for_each(|(entry, value)| {
                        entry.matrix_map.insert(name.to_string(), value);
                    });

                self.fields.insert(
                    name.to_string(),
                    FieldType {
                        datatype: DataType::Matrix,
                        prunable,
                    },
                );
                Ok(())
            }
        }
    }

    #[anyinput]
    pub fn add_matrix_field_par(
        &mut self,
        name: AnyString,
        values: Vec<Matrix64>,
        prunable: bool,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        match self.fields.get(name) {
            Some(ft) => Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
                data_type: ft.datatype,
            }),
            None => {
                self.entries
                    .par_iter_mut()
                    .zip(values.into_par_iter())
                    .for_each(|(entry, value)| {
                        entry.matrix_map.insert(name.to_string(), value);
                    });

                self.fields.insert(
                    name.to_string(),
                    FieldType {
                        datatype: DataType::Matrix,
                        prunable,
                    },
                );
                Ok(())
            }
        }
    }

    #[anyinput]
    pub fn add_cmatrix_field(
        &mut self,
        name: AnyString,
        values: Vec<CMatrix64>,
        prunable: bool,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        match self.fields.get(name) {
            Some(ft) => Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
                data_type: ft.datatype,
            }),
            None => {
                self.entries
                    .iter_mut()
                    .zip(values.into_iter())
                    .for_each(|(entry, value)| {
                        entry.cmatrix_map.insert(name.to_string(), value);
                    });

                self.fields.insert(
                    name.to_string(),
                    FieldType {
                        datatype: DataType::CMatrix,
                        prunable,
                    },
                );
                Ok(())
            }
        }
    }

    #[anyinput]
    pub fn add_cmatrix_field_par(
        &mut self,
        name: AnyString,
        values: Vec<CMatrix64>,
        prunable: bool,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        match self.fields.get(name) {
            Some(ft) => Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
                data_type: ft.datatype,
            }),
            None => {
                self.entries
                    .par_iter_mut()
                    .zip(values.into_par_iter())
                    .for_each(|(entry, value)| {
                        entry.cmatrix_map.insert(name.to_string(), value);
                    });

                self.fields.insert(
                    name.to_string(),
                    FieldType {
                        datatype: DataType::CMatrix,
                        prunable,
                    },
                );
                Ok(())
            }
        }
    }

    #[anyinput]
    pub fn add_momentum_field(
        &mut self,
        name: AnyString,
        values: Vec<Momentum64>,
        prunable: bool,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        match self.fields.get(name) {
            Some(ft) => Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
                data_type: ft.datatype,
            }),
            None => {
                self.entries
                    .iter_mut()
                    .zip(values.into_iter())
                    .for_each(|(entry, value)| {
                        entry.momentum_map.insert(name.to_string(), value);
                    });

                self.fields.insert(
                    name.to_string(),
                    FieldType {
                        datatype: DataType::Momentum,
                        prunable,
                    },
                );
                Ok(())
            }
        }
    }

    #[anyinput]
    pub fn add_momentum_field_par(
        &mut self,
        name: AnyString,
        values: Vec<Momentum64>,
        prunable: bool,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        match self.fields.get(name) {
            Some(ft) => Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
                data_type: ft.datatype,
            }),
            None => {
                self.entries
                    .par_iter_mut()
                    .zip(values.into_par_iter())
                    .for_each(|(entry, value)| {
                        entry.momentum_map.insert(name.to_string(), value);
                    });

                self.fields.insert(
                    name.to_string(),
                    FieldType {
                        datatype: DataType::Momentum,
                        prunable,
                    },
                );
                Ok(())
            }
        }
    }

    #[anyinput]
    pub fn add_momenta_field(
        &mut self,
        name: AnyString,
        values: Vec<Momenta64>,
        prunable: bool,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        match self.fields.get(name) {
            Some(ft) => Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
                data_type: ft.datatype,
            }),
            None => {
                self.entries
                    .iter_mut()
                    .zip(values.into_iter())
                    .for_each(|(entry, value)| {
                        entry.momenta_map.insert(name.to_string(), value);
                    });

                self.fields.insert(
                    name.to_string(),
                    FieldType {
                        datatype: DataType::MomentumVector,
                        prunable,
                    },
                );
                Ok(())
            }
        }
    }

    #[anyinput]
    pub fn add_momenta_field_par(
        &mut self,
        name: AnyString,
        values: Vec<Momenta64>,
        prunable: bool,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        match self.fields.get(name) {
            Some(ft) => Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
                data_type: ft.datatype,
            }),
            None => {
                self.entries
                    .par_iter_mut()
                    .zip(values.into_par_iter())
                    .for_each(|(entry, value)| {
                        entry.momenta_map.insert(name.to_string(), value);
                    });

                self.fields.insert(
                    name.to_string(),
                    FieldType {
                        datatype: DataType::MomentumVector,
                        prunable,
                    },
                );
                Ok(())
            }
        }
    }

    pub fn prune(&mut self) {
        let mut p_scalar: Vec<String> = Vec::with_capacity(self.fields.len());
        let mut p_cscalar: Vec<String> = Vec::with_capacity(self.fields.len());
        let mut p_vector: Vec<String> = Vec::with_capacity(self.fields.len());
        let mut p_cvector: Vec<String> = Vec::with_capacity(self.fields.len());
        let mut p_matrix: Vec<String> = Vec::with_capacity(self.fields.len());
        let mut p_cmatrix: Vec<String> = Vec::with_capacity(self.fields.len());
        let mut p_momentum: Vec<String> = Vec::with_capacity(self.fields.len());
        let mut p_momenta: Vec<String> = Vec::with_capacity(self.fields.len());
        for (name, FieldType { datatype, prunable }) in self.fields.iter() {
            if *prunable {
                match datatype {
                    DataType::Scalar => p_scalar.push(name.to_string()),
                    DataType::CScalar => p_cscalar.push(name.to_string()),
                    DataType::Vector => p_vector.push(name.to_string()),
                    DataType::CVector => p_cvector.push(name.to_string()),
                    DataType::Matrix => p_matrix.push(name.to_string()),
                    DataType::CMatrix => p_cmatrix.push(name.to_string()),
                    DataType::Momentum => p_momentum.push(name.to_string()),
                    DataType::MomentumVector => p_momenta.push(name.to_string()),
                }
            }
        }
        self.entries.iter_mut().for_each(|entry| {
            for key in &p_scalar {
                entry.scalar_map.remove(key);
                self.fields.remove(key);
            }
            for key in &p_cscalar {
                entry.cscalar_map.remove(key);
                self.fields.remove(key);
            }
            for key in &p_vector {
                entry.vector_map.remove(key);
                self.fields.remove(key);
            }
            for key in &p_cvector {
                entry.cvector_map.remove(key);
                self.fields.remove(key);
            }
            for key in &p_matrix {
                entry.matrix_map.remove(key);
                self.fields.remove(key);
            }
            for key in &p_cmatrix {
                entry.cmatrix_map.remove(key);
                self.fields.remove(key);
            }
            for key in &p_momentum {
                entry.momentum_map.remove(key);
                self.fields.remove(key);
            }
            for key in &p_momenta {
                entry.momenta_map.remove(key);
                self.fields.remove(key);
            }
        })
    }

    pub fn prune_par(&mut self) {
        let mut p_scalar: Vec<String> = Vec::with_capacity(self.fields.len());
        let mut p_cscalar: Vec<String> = Vec::with_capacity(self.fields.len());
        let mut p_vector: Vec<String> = Vec::with_capacity(self.fields.len());
        let mut p_cvector: Vec<String> = Vec::with_capacity(self.fields.len());
        let mut p_matrix: Vec<String> = Vec::with_capacity(self.fields.len());
        let mut p_cmatrix: Vec<String> = Vec::with_capacity(self.fields.len());
        let mut p_momentum: Vec<String> = Vec::with_capacity(self.fields.len());
        let mut p_momenta: Vec<String> = Vec::with_capacity(self.fields.len());
        let mut p_all: Vec<String> = Vec::with_capacity(self.fields.len());
        for (name, FieldType { datatype, prunable }) in self.fields.iter() {
            if *prunable {
                match datatype {
                    DataType::Scalar => p_scalar.push(name.to_string()),
                    DataType::CScalar => p_cscalar.push(name.to_string()),
                    DataType::Vector => p_vector.push(name.to_string()),
                    DataType::CVector => p_cvector.push(name.to_string()),
                    DataType::Matrix => p_matrix.push(name.to_string()),
                    DataType::CMatrix => p_cmatrix.push(name.to_string()),
                    DataType::Momentum => p_momentum.push(name.to_string()),
                    DataType::MomentumVector => p_momenta.push(name.to_string()),
                }
                p_all.push(name.to_string());
            }
        }
        for name in p_all {
            self.fields.remove(&name);
        }
        self.entries.par_iter_mut().for_each(|entry| {
            for key in &p_scalar {
                entry.scalar_map.remove(key);
            }
            for key in &p_cscalar {
                entry.cscalar_map.remove(key);
            }
            for key in &p_vector {
                entry.vector_map.remove(key);
            }
            for key in &p_cvector {
                entry.cvector_map.remove(key);
            }
            for key in &p_matrix {
                entry.matrix_map.remove(key);
            }
            for key in &p_cmatrix {
                entry.cmatrix_map.remove(key);
            }
            for key in &p_momentum {
                entry.momentum_map.remove(key);
            }
            for key in &p_momenta {
                entry.momenta_map.remove(key);
            }
        })
    }

    pub fn scalars_to_momentum(
        e_vec: Vec<Scalar64>,
        px_vec: Vec<Scalar64>,
        py_vec: Vec<Scalar64>,
        pz_vec: Vec<Scalar64>,
    ) -> Vec<Momentum64> {
        e_vec
            .into_iter()
            .zip(px_vec.into_iter())
            .zip(py_vec.into_iter())
            .zip(pz_vec.into_iter())
            .map(|(((e, px), py), pz)| FourMomentum::new(e, px, py, pz))
            .collect()
    }

    pub fn scalars_to_momentum_par(
        e_vec: Vec<Scalar64>,
        px_vec: Vec<Scalar64>,
        py_vec: Vec<Scalar64>,
        pz_vec: Vec<Scalar64>,
    ) -> Vec<Momentum64> {
        e_vec
            .into_par_iter()
            .zip(px_vec.into_par_iter())
            .zip(py_vec.into_par_iter())
            .zip(pz_vec.into_par_iter())
            .map(|(((e, px), py), pz)| FourMomentum::new(e, px, py, pz))
            .collect()
    }

    pub fn vectors_to_momenta(
        es_vec: Vec<Vector64>,
        pxs_vec: Vec<Vector64>,
        pys_vec: Vec<Vector64>,
        pzs_vec: Vec<Vector64>,
    ) -> Vec<Momenta64> {
        es_vec
            .into_iter()
            .zip(pxs_vec.into_iter())
            .zip(pys_vec.into_iter())
            .zip(pzs_vec.into_iter())
            .map(|(((es, pxs), pys), pzs)| {
                es.into_iter()
                    .zip(pxs.into_iter())
                    .zip(pys.into_iter())
                    .zip(pzs.into_iter())
                    .map(|(((e, px), py), pz)| FourMomentum::new(e, px, py, pz))
                    .collect()
            })
            .collect()
    }

    pub fn vectors_to_momenta_par(
        es_vec: Vec<Vector64>,
        pxs_vec: Vec<Vector64>,
        pys_vec: Vec<Vector64>,
        pzs_vec: Vec<Vector64>,
    ) -> Vec<Momenta64> {
        es_vec
            .into_par_iter()
            .zip(pxs_vec.into_par_iter())
            .zip(pys_vec.into_par_iter())
            .zip(pzs_vec.into_par_iter())
            .map(|(((es, pxs), pys), pzs)| {
                es.into_iter()
                    .zip(pxs.into_iter())
                    .zip(pys.into_iter())
                    .zip(pzs.into_iter())
                    .map(|(((e, px), py), pz)| FourMomentum::new(e, px, py, pz))
                    .collect()
            })
            .collect()
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

#[derive(Clone, Copy, Debug, Variantly)]
pub enum ReadType {
    F32,
    F64,
}

#[anyinput]
pub fn extract_scalar(
    column_name: AnyString,
    dataframe: &DataFrame,
    read_type: ReadType,
) -> Vec<Scalar64> {
    //! Extract a Scalar64 field from a [`polars`] [`Dataset`]
    //!
    //! # Panics
    //! This function will panic if the column name is invalid or if the assumed types are
    //! incorrect
    let series = dataframe.column(column_name).unwrap();
    match read_type {
        ReadType::F32 => series
            .f32()
            .unwrap()
            .to_vec()
            .into_iter()
            .collect::<Option<Vec<f32>>>()
            .unwrap()
            .into_iter()
            .map(|val| val as Scalar64)
            .collect::<Vec<Scalar64>>(),
        ReadType::F64 => series
            .f64()
            .unwrap()
            .to_vec()
            .into_iter()
            .collect::<Option<Vec<f64>>>()
            .unwrap()
            .into_iter()
            .collect::<Vec<Scalar64>>(),
    }
}

#[anyinput]
pub fn extract_vector(
    column_name: AnyString,
    dataframe: &DataFrame,
    read_type: ReadType,
) -> Vec<Vector64> {
    //! Extract a Vector64 field from a [`polars`] [`Dataset`]
    //!
    //! # Panics
    //! This function will panic if the column name is invalid or if the assumed types are
    //! incorrect
    let series = dataframe.column(column_name).unwrap();
    let vec_of_subseries = series
        .list()
        .unwrap()
        .into_iter()
        .collect::<Option<Vec<Series>>>()
        .unwrap();
    match read_type {
        ReadType::F32 => vec_of_subseries
            .into_iter()
            .map(|subseries| {
                subseries
                    .f32()
                    .unwrap()
                    .into_iter()
                    .map(|val| val.unwrap() as f64)
                    .collect::<Vector64>()
            })
            .collect::<Vec<Vector64>>(),
        ReadType::F64 => vec_of_subseries
            .into_iter()
            .map(|subseries| {
                subseries
                    .f64()
                    .unwrap()
                    .into_iter()
                    .map(|val| val.unwrap())
                    .collect::<Vector64>()
            })
            .collect::<Vec<Vector64>>(),
    }
}
