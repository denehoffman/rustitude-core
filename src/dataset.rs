use anyinput::anyinput;
use polars::prelude::*;
use rayon::prelude::*;
use uuid::Uuid;

use crate::prelude::FourMomentum;

pub trait Event {}
pub struct Dataset<T>
where
    T: Event,
{
    pub uuid: Uuid,
    pub events: Vec<T>,
}

impl<T> Dataset<T>
where
    T: Event,
{
    pub fn new() -> Self {
        Dataset {
            uuid: Uuid::new_v4(),
            events: Vec::new(),
        }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.events.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.events.iter_mut()
    }
}
impl<T> Default for Dataset<T>
where
    T: Event,
{
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ReadType {
    F32,
    F64,
}

#[anyinput]
pub fn extract_scalar(
    column_name: AnyString,
    dataframe: &DataFrame,
    read_type: ReadType,
) -> Vec<f64> {
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
            .map(|val| val as f64)
            .collect::<Vec<f64>>(),
        ReadType::F64 => series
            .f64()
            .unwrap()
            .to_vec()
            .into_iter()
            .collect::<Option<Vec<f64>>>()
            .unwrap()
            .into_iter()
            .collect::<Vec<f64>>(),
    }
}

#[anyinput]
pub fn extract_vector(
    column_name: AnyString,
    dataframe: &DataFrame,
    read_type: ReadType,
) -> Vec<Vec<f64>> {
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
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>(),
        ReadType::F64 => vec_of_subseries
            .into_iter()
            .map(|subseries| {
                subseries
                    .f64()
                    .unwrap()
                    .into_iter()
                    .map(|val| val.unwrap())
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>(),
    }
}

pub fn scalars_to_momentum(
    e_vec: Vec<f64>,
    px_vec: Vec<f64>,
    py_vec: Vec<f64>,
    pz_vec: Vec<f64>,
) -> Vec<FourMomentum> {
    e_vec
        .into_par_iter()
        .zip(px_vec.into_par_iter())
        .zip(py_vec.into_par_iter())
        .zip(pz_vec.into_par_iter())
        .map(|(((e, px), py), pz)| vec![e, px, py, pz].into())
        .collect()
}

pub fn vectors_to_momenta(
    es_vec: Vec<Vec<f64>>,
    pxs_vec: Vec<Vec<f64>>,
    pys_vec: Vec<Vec<f64>>,
    pzs_vec: Vec<Vec<f64>>,
) -> Vec<Vec<FourMomentum>> {
    (es_vec, pxs_vec, pys_vec, pzs_vec)
        .into_par_iter()
        .map(|(es, pxs, pys, pzs)| {
            (es, pxs, pys, pzs)
                .into_par_iter()
                .map(|(e, px, py, pz)| FourMomentum::new(e, px, py, pz))
                .collect()
        })
        .collect()
}
