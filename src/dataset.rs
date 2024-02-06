use std::fs::File;

use polars::prelude::*;
use rayon::prelude::*;

use ndarray::{s, Array1, Array2, Array3, Axis, Zip};
use num_complex::Complex64;
use rustc_hash::FxHashMap as HashMap;

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
}

pub type Scalar64 = f64;
pub type CScalar64 = Complex64;
pub type Vector64 = Array1<f64>;
pub type CVector64 = Array1<Complex64>;
pub type Matrix64 = Array2<f64>;
pub type CMatrix64 = Array2<Complex64>;

#[derive(Default, Debug, Clone)]
pub struct Dataset {
    size: usize,
    weights: Vec<Scalar64>,
    scalar_map: HashMap<String, Vec<Scalar64>>,
    cscalar_map: HashMap<String, Vec<CScalar64>>,
    vector_map: HashMap<String, Vec<Vector64>>,
    cvector_map: HashMap<String, Vec<CVector64>>,
    matrix_map: HashMap<String, Vec<Matrix64>>,
    cmatrix_map: HashMap<String, Vec<CMatrix64>>,
}

#[derive(Error, Debug)]
pub enum DatasetError {
    #[error("No {data_type:?} exists with name {key_name:?}")]
    TypeError {
        key_name: String,
        data_type: DataType,
    },
    #[error("{key_name:?} already exists")]
    FieldExistsError { key_name: String },
    #[error("Cannot store a field of size {field_size} in a dataset of size {dataset_size}")]
    IncompatibleSizesError {
        dataset_size: usize,
        field_size: usize,
    },
}

impl Dataset {
    pub fn from_size(size: usize, weights: Option<Vec<Scalar64>>) -> Dataset {
        match weights {
            Some(w) => Dataset {
                size,
                weights: w,
                ..Default::default()
            },
            None => Dataset {
                size,
                weights: vec![1.0; size],
                ..Default::default()
            },
        }
    }
    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn weights(&self) -> &Vec<Scalar64> {
        &self.weights
    }

    pub fn weighted_len(&self) -> f64 {
        self.weights.iter().sum()
    }

    #[anyinput]
    pub fn scalar(&self, key: AnyString) -> Result<&Vec<Scalar64>, DatasetError> {
        self.scalar_map.get(key).ok_or(DatasetError::TypeError {
            key_name: key.to_string(),
            data_type: DataType::Scalar,
        })
    }

    #[anyinput]
    pub fn cscalar(&self, key: AnyString) -> Result<&Vec<CScalar64>, DatasetError> {
        self.cscalar_map.get(key).ok_or(DatasetError::TypeError {
            key_name: key.to_string(),
            data_type: DataType::CScalar,
        })
    }

    #[anyinput]
    pub fn vector(&self, key: AnyString) -> Result<&Vec<Vector64>, DatasetError> {
        self.vector_map.get(key).ok_or(DatasetError::TypeError {
            key_name: key.to_string(),
            data_type: DataType::Vector,
        })
    }

    #[anyinput]
    pub fn cvector(&self, key: AnyString) -> Result<&Vec<CVector64>, DatasetError> {
        self.cvector_map.get(key).ok_or(DatasetError::TypeError {
            key_name: key.to_string(),
            data_type: DataType::CVector,
        })
    }

    #[anyinput]
    pub fn matrix(&self, key: AnyString) -> Result<&Vec<Matrix64>, DatasetError> {
        self.matrix_map.get(key).ok_or(DatasetError::TypeError {
            key_name: key.to_string(),
            data_type: DataType::Matrix,
        })
    }

    #[anyinput]
    pub fn cmatrix(&self, key: AnyString) -> Result<&Vec<CMatrix64>, DatasetError> {
        self.cmatrix_map.get(key).ok_or(DatasetError::TypeError {
            key_name: key.to_string(),
            data_type: DataType::CMatrix,
        })
    }

    #[anyinput]
    pub fn contains_scalar(&self, key: AnyString) -> bool {
        self.scalar_map.contains_key(key)
    }

    #[anyinput]
    pub fn contains_cscalar(&self, key: AnyString) -> bool {
        self.cscalar_map.contains_key(key)
    }

    #[anyinput]
    pub fn contains_vector(&self, key: AnyString) -> bool {
        self.vector_map.contains_key(key)
    }

    #[anyinput]
    pub fn contains_cvector(&self, key: AnyString) -> bool {
        self.cvector_map.contains_key(key)
    }

    #[anyinput]
    pub fn contains_matrix(&self, key: AnyString) -> bool {
        self.matrix_map.contains_key(key)
    }

    #[anyinput]
    pub fn contains_cmatrix(&self, key: AnyString) -> bool {
        self.cmatrix_map.contains_key(key)
    }

    #[anyinput]
    pub fn insert_scalar(
        &mut self,
        name: AnyString,
        values: Vec<Scalar64>,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        if self.scalar_map.contains_key(name) {
            return Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
            });
        }
        self.scalar_map.insert(name.to_string(), values);
        Ok(())
    }

    #[anyinput]
    pub fn insert_cscalar(
        &mut self,
        name: AnyString,
        values: Vec<CScalar64>,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        if self.cscalar_map.contains_key(name) {
            return Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
            });
        }
        self.cscalar_map.insert(name.to_string(), values);
        Ok(())
    }

    #[anyinput]
    pub fn insert_vector(
        &mut self,
        name: AnyString,
        values: Vec<Vector64>,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        if self.vector_map.contains_key(name) {
            return Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
            });
        }
        self.vector_map.insert(name.to_string(), values);
        Ok(())
    }

    #[anyinput]
    pub fn insert_cvector(
        &mut self,
        name: AnyString,
        values: Vec<CVector64>,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        if self.cvector_map.contains_key(name) {
            return Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
            });
        }
        self.cvector_map.insert(name.to_string(), values);
        Ok(())
    }

    #[anyinput]
    pub fn insert_matrix(
        &mut self,
        name: AnyString,
        values: Vec<Matrix64>,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        if self.matrix_map.contains_key(name) {
            return Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
            });
        }
        self.matrix_map.insert(name.to_string(), values);
        Ok(())
    }

    #[anyinput]
    pub fn insert_cmatrix(
        &mut self,
        name: AnyString,
        values: Vec<CMatrix64>,
    ) -> Result<(), DatasetError> {
        if self.len() != values.len() {
            return Err(DatasetError::IncompatibleSizesError {
                dataset_size: self.len(),
                field_size: values.len(),
            });
        }
        if self.cmatrix_map.contains_key(name) {
            return Err(DatasetError::FieldExistsError {
                key_name: name.to_string(),
            });
        }
        self.cmatrix_map.insert(name.to_string(), values);
        Ok(())
    }
}
pub fn scalars_to_momentum(
    e_vec: Vec<Scalar64>,
    px_vec: Vec<Scalar64>,
    py_vec: Vec<Scalar64>,
    pz_vec: Vec<Scalar64>,
) -> Vec<Vector64> {
    e_vec
        .into_iter()
        .zip(px_vec)
        .zip(py_vec)
        .zip(pz_vec)
        .map(|(((e, px), py), pz)| Array1::from_vec(vec![e, px, py, pz]))
        .collect()
}

pub fn scalars_to_momentum_par(
    e_vec: Vec<Scalar64>,
    px_vec: Vec<Scalar64>,
    py_vec: Vec<Scalar64>,
    pz_vec: Vec<Scalar64>,
) -> Vec<Vector64> {
    e_vec
        .into_par_iter()
        .zip(px_vec.into_par_iter())
        .zip(py_vec.into_par_iter())
        .zip(pz_vec.into_par_iter())
        .map(|(((e, px), py), pz)| Array1::from_vec(vec![e, px, py, pz]))
        .collect()
}

pub fn vectors_to_momenta(
    es_vec: Vec<Vector64>,
    pxs_vec: Vec<Vector64>,
    pys_vec: Vec<Vector64>,
    pzs_vec: Vec<Vector64>,
) -> Vec<Vec<Vector64>> {
    let data = [es_vec, pxs_vec, pys_vec, pzs_vec]; // (component, event, particle)
    let shape = data[0][0].shape();
    let dim = (data.len(), data[0].len(), shape[0]);
    let mut array3 = Array3::zeros(dim);
    for (i, component) in data.iter().enumerate() {
        for (j, event) in component.iter().enumerate() {
            assert_eq!(event.shape(), shape, "Array mismatch!");
            Zip::from(&mut array3.slice_mut(s![i, j, ..]))
                .and(event)
                .for_each(|a3, &a1| *a3 = a1);
        }
    }
    array3.swap_axes(0, 2); // (particle, event, component)
    let shape = array3.shape();
    (0..shape[0]) // 0 - n_particles
        .map(|i| {
            (0..shape[1]) // 0 - n_events
                .map(|j| {
                    array3
                        .index_axis(Axis(0), i)
                        .index_axis(Axis(0), j)
                        .to_owned()
                })
                .collect()
        })
        .collect()
}

pub fn vectors_to_momenta_par(
    es_vec: Vec<Vector64>,
    pxs_vec: Vec<Vector64>,
    pys_vec: Vec<Vector64>,
    pzs_vec: Vec<Vector64>,
) -> Vec<Vec<Vector64>> {
    let data = [es_vec, pxs_vec, pys_vec, pzs_vec]; // (component, event, particle)
    let shape = data[0][0].shape();
    let dim = (data.len(), data[0].len(), shape[0]);
    let mut array3 = Array3::zeros(dim);
    for (i, component) in data.iter().enumerate() {
        for (j, event) in component.iter().enumerate() {
            assert_eq!(event.shape(), shape, "Array mismatch!");
            Zip::from(&mut array3.slice_mut(s![i, j, ..]))
                .and(event)
                .for_each(|a3, &a1| *a3 = a1);
        }
    }
    array3.swap_axes(0, 2); // (particle, event, component)
    let shape = array3.shape();
    (0..shape[0]) // 0 - n_particles
        .into_par_iter()
        .map(|i| {
            (0..shape[1]) // 0 - n_events
                .into_par_iter()
                .map(|j| {
                    array3
                        .index_axis(Axis(0), i)
                        .index_axis(Axis(0), j)
                        .to_owned()
                })
                .collect()
        })
        .collect()
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
