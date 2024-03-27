use rayon::prelude::*;

pub trait Event: Sync + Send {}

#[derive(Default)]
pub struct Dataset<T: Event>(Vec<T>);

impl<T> Dataset<T>
where
    T: Event,
{
    pub fn new(data: Vec<T>) -> Self {
        Dataset(data)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.0.iter_mut()
    }

    pub fn par_iter(&self) -> rayon::slice::Iter<'_, T> {
        self.0.par_iter()
    }

    pub fn par_iter_mut(&mut self) -> rayon::slice::IterMut<'_, T> {
        self.0.par_iter_mut()
    }
}
