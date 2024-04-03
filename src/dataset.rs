use std::{fmt::Display, fs::File, path::Path};

use nalgebra::Vector3;
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::{Field, Row},
};
use rayon::prelude::*;

use crate::prelude::FourMomentum;

#[derive(Debug, Default)]
pub struct Event {
    pub index: usize,
    pub weight: f64,
    pub beam_p4: FourMomentum,
    pub recoil_p4: FourMomentum,
    pub daughter_p4s: Vec<FourMomentum>,
    pub eps: Vector3<f64>,
}

impl Display for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Index: {}", self.index)?;
        writeln!(f, "Weight: {}", self.weight)?;
        writeln!(f, "Beam P4: {}", self.beam_p4)?;
        writeln!(f, "Recoil P4: {}", self.recoil_p4)?;
        writeln!(f, "Daughters:")?;
        for (i, p4) in self.daughter_p4s.iter().enumerate() {
            writeln!(f, "\t{i} -> {p4}")?;
        }
        writeln!(
            f,
            "EPS: [{}, {}, {}]",
            self.eps[0], self.eps[1], self.eps[2]
        )?;
        Ok(())
    }
}
impl Event {
    fn read(index: usize, row: Row, polarized: bool) -> Self {
        let mut event = Event {
            index,
            ..Default::default()
        };
        let mut e_fs: Vec<f64> = Vec::new();
        let mut px_fs: Vec<f64> = Vec::new();
        let mut py_fs: Vec<f64> = Vec::new();
        let mut pz_fs: Vec<f64> = Vec::new();
        for (name, field) in row.get_column_iter() {
            match (name.as_str(), field) {
                ("E_Beam", Field::Float(value)) => {
                    event.beam_p4.set_e(f64::from(*value));
                }
                ("Px_Beam", Field::Float(value)) => {
                    if polarized {
                        event.eps[0] = f64::from(*value);
                    } else {
                        event.beam_p4.set_px(f64::from(*value));
                    }
                }
                ("Py_Beam", Field::Float(value)) => {
                    if polarized {
                        event.eps[1] = f64::from(*value);
                    } else {
                        event.beam_p4.set_py(f64::from(*value));
                    }
                }
                ("Pz_Beam", Field::Float(value)) => {
                    event.beam_p4.set_pz(f64::from(*value));
                }
                ("Weight", Field::Float(value)) => event.weight = f64::from(*value),
                ("E_FinalState", Field::ListInternal(list)) => {
                    e_fs = list
                        .elements()
                        .iter()
                        .map(|field| {
                            if let Field::Float(value) = field {
                                f64::from(*value)
                            } else {
                                panic!()
                            }
                        })
                        .collect()
                }
                ("Px_FinalState", Field::ListInternal(list)) => {
                    px_fs = list
                        .elements()
                        .iter()
                        .map(|field| {
                            if let Field::Float(value) = field {
                                f64::from(*value)
                            } else {
                                panic!()
                            }
                        })
                        .collect()
                }
                ("Py_FinalState", Field::ListInternal(list)) => {
                    py_fs = list
                        .elements()
                        .iter()
                        .map(|field| {
                            if let Field::Float(value) = field {
                                f64::from(*value)
                            } else {
                                panic!()
                            }
                        })
                        .collect()
                }
                ("Pz_FinalState", Field::ListInternal(list)) => {
                    pz_fs = list
                        .elements()
                        .iter()
                        .map(|field| {
                            if let Field::Float(value) = field {
                                f64::from(*value)
                            } else {
                                panic!()
                            }
                        })
                        .collect()
                }
                _ => {}
            }
        }
        event.recoil_p4 = FourMomentum::new(e_fs[0], px_fs[0], py_fs[0], pz_fs[0]);
        event.daughter_p4s = e_fs[1..]
            .iter()
            .zip(px_fs[1..].iter())
            .zip(py_fs[1..].iter())
            .zip(pz_fs[1..].iter())
            .map(|(((e, px), py), pz)| FourMomentum::new(*e, *px, *py, *pz))
            .collect();
        let final_state_p4 = event.recoil_p4 + event.daughter_p4s.iter().sum();
        event.beam_p4 = event.beam_p4.boost_along(&final_state_p4);
        event.recoil_p4 = event.recoil_p4.boost_along(&final_state_p4);
        for dp4 in event.daughter_p4s.iter_mut() {
            *dp4 = dp4.boost_along(&final_state_p4);
        }
        event
    }
}

#[derive(Default)]
pub struct Dataset {
    pub events: Vec<Event>,
}

impl Dataset {
    pub fn new(events: Vec<Event>) -> Self {
        Dataset { events }
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Event> {
        self.events.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Event> {
        self.events.iter_mut()
    }

    pub fn par_iter(&self) -> rayon::slice::Iter<'_, Event> {
        self.events.par_iter()
    }

    pub fn par_iter_mut(&mut self) -> rayon::slice::IterMut<'_, Event> {
        self.events.par_iter_mut()
    }

    pub fn from_parquet(path: &str, polarized: bool) -> Dataset {
        let path = Path::new(path);
        let file = File::open(path).unwrap();
        let reader = SerializedFileReader::new(file).unwrap();
        let row_iter = reader.get_row_iter(None).unwrap();
        Dataset::new(
            row_iter
                .enumerate()
                .map(|(i, row)| Event::read(i, row.unwrap(), polarized))
                .collect(),
        )
    }
}
