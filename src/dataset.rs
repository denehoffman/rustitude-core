use uuid::Uuid;

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
