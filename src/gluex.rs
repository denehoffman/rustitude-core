use std::{f64::consts::PI, fmt::Display, fs::File, path::Path};

use crate::{
    amplitude::ParameterSet, dataset::{Dataset, Event}, prelude::{Amplitude, FourMomentum}
};

use nalgebra::{ComplexField, SMatrix, SVector, Vector3};
use num_complex::Complex64;
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::{Field, Row},
};
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use sphrs::{ComplexSH, Coordinates, SHEval};
use uuid::Uuid;

pub trait Weight {
    fn weight(&self) -> &f64;
}

pub trait BeamP4 {
    fn beam_p4(&self) -> &FourMomentum;
}

pub trait RecoilP4 {
    fn recoil_p4(&self) -> &FourMomentum;
}

pub trait DaughterP4s {
    fn daughter_p4s(&self) -> &Vec<FourMomentum>;
}

pub trait Polarized {
    fn eps(&self) -> &Vector3<f64>;
}

#[derive(Debug, Default)]
pub struct GlueXEvent {
    weight: f64,
    beam_p4: FourMomentum,
    recoil_p4: FourMomentum,
    daughter_p4s: Vec<FourMomentum>,
}
impl GlueXEvent {
    fn ensure_com_boost(&mut self) {
        //! Boosts everything to the Center-Of-Mass frame (things already in the COM frame will be
        //! unchanged by this transformation)
        let final_state_p4 = self.recoil_p4 + self.daughter_p4s.iter().sum();
        self.beam_p4 = self.beam_p4.boost_along(&final_state_p4);
        self.recoil_p4 = self.recoil_p4.boost_along(&final_state_p4);
        for dp4 in self.daughter_p4s.iter_mut() {
            *dp4 = dp4.boost_along(&final_state_p4);
        }
    }
}
impl Event for GlueXEvent {}
impl Weight for GlueXEvent {
    fn weight(&self) -> &f64 {
        &self.weight
    }
}
impl BeamP4 for GlueXEvent {
    fn beam_p4(&self) -> &FourMomentum {
        &self.beam_p4
    }
}
impl RecoilP4 for GlueXEvent {
    fn recoil_p4(&self) -> &FourMomentum {
        &self.recoil_p4
    }
}
impl DaughterP4s for GlueXEvent {
    fn daughter_p4s(&self) -> &Vec<FourMomentum> {
        &self.daughter_p4s
    }
}
impl Display for GlueXEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Weight: {}", self.weight)?;
        writeln!(f, "Beam P4: {}", self.beam_p4)?;
        writeln!(f, "Recoil P4: {}", self.recoil_p4)?;
        writeln!(f, "Daughters:")?;
        for (i, p4) in self.daughter_p4s.iter().enumerate() {
            writeln!(f, "\t{i} -> {p4}")?;
        }
        Ok(())
    }
}
impl From<Row> for GlueXEvent {
    fn from(row: Row) -> Self {
        let mut event = GlueXEvent::default();
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
                    event.beam_p4.set_px(f64::from(*value));
                }
                ("Py_Beam", Field::Float(value)) => {
                    event.beam_p4.set_py(f64::from(*value));
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
        event.ensure_com_boost();
        event
    }
}
impl From<GlueXPolarizedEvent> for GlueXEvent {
    fn from(event: GlueXPolarizedEvent) -> Self {
        GlueXEvent {
            weight: event.weight,
            beam_p4: event.beam_p4,
            recoil_p4: event.recoil_p4,
            daughter_p4s: event.daughter_p4s,
        }
    }
}

#[derive(Debug, Default)]
pub struct GlueXPolarizedEvent {
    weight: f64,
    beam_p4: FourMomentum,
    recoil_p4: FourMomentum,
    daughter_p4s: Vec<FourMomentum>,
    eps: Vector3<f64>,
}
impl GlueXPolarizedEvent {
    fn ensure_com_boost(&mut self) {
        //! Boosts everything to the Center-Of-Mass frame (things already in the COM frame will be
        //! unchanged by this transformation)
        let final_state_p4 = self.recoil_p4 + self.daughter_p4s.iter().sum();
        self.beam_p4 = self.beam_p4.boost_along(&final_state_p4);
        self.recoil_p4 = self.recoil_p4.boost_along(&final_state_p4);
        for dp4 in self.daughter_p4s.iter_mut() {
            *dp4 = dp4.boost_along(&final_state_p4);
        }
    }
}
impl Event for GlueXPolarizedEvent {}
impl Weight for GlueXPolarizedEvent {
    fn weight(&self) -> &f64 {
        &self.weight
    }
}
impl BeamP4 for GlueXPolarizedEvent {
    fn beam_p4(&self) -> &FourMomentum {
        &self.beam_p4
    }
}
impl RecoilP4 for GlueXPolarizedEvent {
    fn recoil_p4(&self) -> &FourMomentum {
        &self.recoil_p4
    }
}
impl DaughterP4s for GlueXPolarizedEvent {
    fn daughter_p4s(&self) -> &Vec<FourMomentum> {
        &self.daughter_p4s
    }
}
impl Polarized for GlueXPolarizedEvent {
    fn eps(&self) -> &Vector3<f64> {
        &self.eps
    }
}
impl Display for GlueXPolarizedEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
impl From<Row> for GlueXPolarizedEvent {
    fn from(row: Row) -> Self {
        let mut event = GlueXPolarizedEvent::default();
        let mut e_fs: Vec<f64> = Vec::new();
        let mut px_fs: Vec<f64> = Vec::new();
        let mut py_fs: Vec<f64> = Vec::new();
        let mut pz_fs: Vec<f64> = Vec::new();
        for (name, field) in row.get_column_iter() {
            match (name.as_str(), field) {
                ("E_Beam", Field::Float(value)) => {
                    event.beam_p4.set_e(f64::from(*value));
                }
                ("Px_Beam", Field::Float(value)) => event.eps[0] = f64::from(*value),
                ("Py_Beam", Field::Float(value)) => event.eps[1] = f64::from(*value),
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
        event.ensure_com_boost();
        event
    }
}

pub fn open_gluex_parquet(path: &str) -> Dataset<GlueXEvent> {
    let path = Path::new(path);
    let file = File::open(path).unwrap();
    let reader = SerializedFileReader::new(file).unwrap();
    let row_iter = reader.get_row_iter(None).unwrap();
    let mut dataset = Dataset::new();
    dataset.events = row_iter.map(|row| GlueXEvent::from(row.unwrap())).collect();
    dataset
}
pub fn open_gluex_pol_in_beam_parquet(path: &str) -> Dataset<GlueXPolarizedEvent> {
    let path = Path::new(path);
    let file = File::open(path).unwrap();
    let reader = SerializedFileReader::new(file).unwrap();
    let row_iter = reader.get_row_iter(None).unwrap();
    let mut dataset = Dataset::new();
    dataset.events = row_iter
        .map(|row| GlueXPolarizedEvent::from(row.unwrap()))
        .collect();
    dataset
}

// NOTE TO SELF: for now we'll assume these are COM boosted, but we can add in a thing to do it
// automatically later, maybe a COM boost function

#[derive(Clone, Copy, Default)]
#[rustfmt::skip]
pub enum Wave {
    #[default]
    S,
    S0,
    Pn1, P0, P1, P,
    Dn2, Dn1, D0, D1, D2, D,
    Fn3, Fn2, Fn1, F0, F1, F2, F3, F,
}

#[rustfmt::skip]
impl Wave {
    pub fn l(&self) -> i64 {
        match self {
            Self::S0 | Self::S => 0,
            Self::Pn1 | Self::P0 | Self::P1 | Self::P => 1,
            Self::Dn2 | Self::Dn1 | Self::D0 | Self::D1 | Self::D2 | Self::D => 2,
            Self::Fn3 | Self::Fn2 | Self::Fn1 | Self::F0 | Self::F1 | Self::F2 | Self::F3 | Self::F => 3,
        }
    }
    pub fn m(&self) -> i64 {
        match self {
            Self::S | Self::P | Self::D | Self::F => 0,
            Self::S0 | Self::P0 | Self::D0 | Self::F0 => 0,
            Self::Pn1 | Self::Dn1 | Self::Fn1 => -1,
            Self::P1 | Self::D1 | Self::F1 => 1,
            Self::Dn2 | Self::Fn2 => -2,
            Self::D2 | Self::F2 => 2,
            Self::Fn3 => -3,
            Self::F3 => 3,
        }
    }
}

impl Display for Wave {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let l_string = match self.l() {
            0 => "S",
            1 => "P",
            2 => "D",
            3 => "F",
            _ => unimplemented!(),
        };
        write!(f, "{} {:+}", l_string, self.m())
    }
}

#[derive(Default)]
pub struct Ylm {
    wave: Wave,
    data: HashMap<Uuid, Vec<Complex64>>,
}
impl Ylm {
    pub fn new(wave: Wave) -> Self {
        Ylm {
            wave,
            ..Default::default()
        }
    }
}
impl<T: Event + BeamP4 + RecoilP4 + DaughterP4s + Sync> Amplitude<T> for Ylm {
    fn calculate(
        &mut self,
        _parameters: &ParameterSet,
        dataset: &Dataset<T>,
    ) -> Vec<Complex64> {
        let precalcs = self.data.entry(dataset.uuid).or_insert_with(|| {
            dataset
                .events
                .par_iter()
                .map(|event| {
                    let resonance = event.daughter_p4s()[0] + event.daughter_p4s()[1];
                    let p1 = event.daughter_p4s()[0];
                    let recoil_res = event.recoil_p4().boost_along(&resonance);
                    let p1_res = p1.boost_along(&resonance);
                    let z = -1.0 * recoil_res.momentum().normalize();
                    let y = event
                        .beam_p4()
                        .momentum()
                        .cross(&(-1.0 * event.recoil_p4().momentum()));
                    let x = y.cross(&z);
                    let p1_vec = p1_res.momentum();
                    let p = Coordinates::cartesian(p1_vec.dot(&x), p1_vec.dot(&y), p1_vec.dot(&z));
                    ComplexSH::Spherical.eval(self.wave.l(), self.wave.m(), &p)
            }).collect()
        });
        precalcs.to_vec()
    }
}

#[derive(Default)]
pub enum Reflectivity {
    #[default]
    Positive,
    Negative,
}

impl Display for Reflectivity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Reflectivity::Positive => write!(f, "+"),
            Reflectivity::Negative => write!(f, "-"),
        }
    }
}

#[derive(Default)]
pub struct Zlm {
    wave: Wave,
    reflectivity: Reflectivity,
    data: HashMap<Uuid, Vec<Complex64>>,
}
impl Zlm {
    pub fn new(wave: Wave, reflectivity: Reflectivity) -> Self {
        Zlm {
            wave,
            reflectivity,
            ..Default::default()
        }
    }
}
impl<T: Event + BeamP4 + RecoilP4 + DaughterP4s + Polarized + Sync> Amplitude<T> for Zlm {
    fn calculate(
        &mut self,
        _parameters: &ParameterSet,
        dataset: &Dataset<T>,
    ) -> Vec<Complex64> {
        let precalcs = self.data.entry(dataset.uuid).or_insert_with(|| {
            dataset
                .events
                .par_iter()
                .map(|event| {
                    let resonance = event.daughter_p4s()[0] + event.daughter_p4s()[1];
                    let p1 = event.daughter_p4s()[0];
                    let recoil_res = event.recoil_p4().boost_along(&resonance);
                    let p1_res = p1.boost_along(&resonance);
                    let z = -1.0 * recoil_res.momentum().normalize();
                    let y = event
                        .beam_p4()
                        .momentum()
                        .cross(&(-1.0 * event.recoil_p4().momentum()));
                    let x = y.cross(&z);
                    let p1_vec = p1_res.momentum();
                    let p = Coordinates::cartesian(p1_vec.dot(&x), p1_vec.dot(&y), p1_vec.dot(&z));
                    let ylm = ComplexSH::Spherical.eval(self.wave.l(), self.wave.m(), &p);
                    let big_phi = y.dot(event.eps()).atan2(
                        event
                            .beam_p4()
                            .momentum()
                            .normalize()
                            .dot(&event.eps().cross(&y)),
                    );
                    let pgamma = event.eps().norm();

                    let phase = Complex64::cis(-big_phi);
                    let zlm = ylm * phase;
                    match self.reflectivity {
                        Reflectivity::Positive => Complex64::new(
                            (1.0 + pgamma).sqrt() * zlm.re,
                            (1.0 - pgamma).sqrt() * zlm.im,
                        ),
                        Reflectivity::Negative => Complex64::new(
                            (1.0 - pgamma).sqrt() * zlm.re,
                            (1.0 + pgamma).sqrt() * zlm.im,
                        ),
                    }
            }).collect()
        });
        precalcs.to_vec()
    }
}


pub fn zlm<T>(event: &T, wave: Wave, reflectivity: Reflectivity) -> Complex64
where
    T: Event + BeamP4 + RecoilP4 + DaughterP4s + Polarized,
{
    let resonance = event.daughter_p4s()[0] + event.daughter_p4s()[1];
    let p1 = event.daughter_p4s()[0];
    let recoil_res = event.recoil_p4().boost_along(&resonance);
    let p1_res = p1.boost_along(&resonance);
    let z = -1.0 * recoil_res.momentum().normalize();
    let y = event
        .beam_p4()
        .momentum()
        .cross(&(-1.0 * event.recoil_p4().momentum()));
    let x = y.cross(&z);
    let p1_vec = p1_res.momentum();
    let p = Coordinates::cartesian(p1_vec.dot(&x), p1_vec.dot(&y), p1_vec.dot(&z));
    let ylm = ComplexSH::Spherical.eval(wave.l(), wave.m(), &p);
    let big_phi = y.dot(event.eps()).atan2(
        event
            .beam_p4()
            .momentum()
            .normalize()
            .dot(&event.eps().cross(&y)),
    );
    let pgamma = event.eps().norm();

    let phase = Complex64::cis(-big_phi);
    let zlm = ylm * phase;
    match reflectivity {
        Reflectivity::Positive => Complex64::new(
            (1.0 + pgamma).sqrt() * zlm.re,
            (1.0 - pgamma).sqrt() * zlm.im,
        ),
        Reflectivity::Negative => Complex64::new(
            (1.0 - pgamma).sqrt() * zlm.re,
            (1.0 + pgamma).sqrt() * zlm.im,
        ),
    }
}

struct KMatrixConstants<const C: usize, const R: usize> {
    g: SMatrix<f64, C, R>,
    c: SMatrix<f64, C, C>,
    m1s: [f64; C],
    m2s: [f64; C],
    mrs: [f64; R],
    adler_zero: Option<AdlerZero>,
    l: usize,
}

fn chi_plus(s: f64, m1: f64, m2: f64) -> f64 {
    1.0 - ((m1 + m2) * (m1 + m2)) / s
}

fn chi_minus(s: f64, m1: f64, m2: f64) -> f64 {
    1.0 - ((m1 - m2) * (m1 - m2)) / s
}

fn rho(s: f64, m1: f64, m2: f64) -> Complex64 {
    Complex64::from(chi_plus(s, m1, m2) * chi_minus(s, m1, m2)).sqrt()
}
fn c_matrix<const C: usize, const R: usize>(
    s: f64,
    constants: &KMatrixConstants<C, R>,
) -> SMatrix<Complex64, C, C> {
    SMatrix::from_diagonal(&SVector::from_fn(|i, _| {
        rho(s, constants.m1s[i], constants.m2s[i]) / PI
            * ((chi_plus(s, constants.m1s[i], constants.m2s[i])
                + rho(s, constants.m1s[i], constants.m2s[i]))
                / (chi_plus(s, constants.m1s[i], constants.m2s[i])
                    - rho(s, constants.m1s[i], constants.m2s[i])))
            .ln()
            + chi_plus(s, constants.m1s[i], constants.m2s[i]) / PI
                * ((constants.m2s[i] - constants.m1s[i]) / (constants.m1s[i] + constants.m2s[i]))
                * (constants.m2s[i] / constants.m1s[i]).ln()
    }))
}
fn z(s: f64, m1: f64, m2: f64) -> Complex64 {
    rho(s, m1, m2).powi(2) * s / (2.0 * 0.1973 * 0.1973)
}
fn blatt_weisskopf(s: f64, m1: f64, m2: f64, l: usize) -> Complex64 {
    let z = z(s, m1, m2);
    match l {
        0 => 1.0.into(),
        1 => ((2.0 * z) / (z + 1.0)).sqrt(),
        2 => ((13.0 * z.powi(2)) / ((z - 3.0).powi(2) + 9.0 * z)).sqrt(),
        3 => {
            ((277.0 * z.powi(3)) / (z * (z - 15.0).powi(2) + 9.0 * (2.0 * z - 5.0).powi(2))).sqrt()
        }
        4 => ((12746.0 * z.powi(4)) / (z.powi(2) - 45.0 * z + 105.0).powi(2)
            + 25.0 * z * (2.0 * z - 21.0).powi(2))
        .sqrt(),
        l => panic!("L = {l} is not yet implemented"),
    }
}
fn barrier_factor(s: f64, m1: f64, m2: f64, mr: f64, l: usize) -> Complex64 {
    blatt_weisskopf(s, m1, m2, l) / blatt_weisskopf(mr.powi(2), m1, m2, l)
}
fn barrier_matrix<const C: usize, const R: usize>(
    s: f64,
    constants: &KMatrixConstants<C, R>,
) -> SMatrix<Complex64, C, R> {
    SMatrix::from_fn(|i, a| {
        barrier_factor(
            s,
            constants.m1s[i],
            constants.m2s[i],
            constants.mrs[a],
            constants.l,
        )
    })
}
#[derive(Clone, Copy)]
pub struct AdlerZero {
    pub s_0: f64,
    pub s_norm: f64,
}
fn k_matrix<const C: usize, const R: usize>(
    s: f64,
    constants: &KMatrixConstants<C, R>,
) -> SMatrix<Complex64, C, C> {
    let bf = barrier_matrix(s, constants);
    SMatrix::from_fn(|i, j| {
        (0..R)
            .map(|a| {
                bf[(i, a)]
                    * bf[(j, a)]
                    * (constants.g[(i, a)] * constants.g[(j, a)] / (constants.mrs[a].powi(2) - s)
                        + constants.c[(i, j)])
            })
            .sum::<Complex64>()
            * constants
                .adler_zero
                .map_or(1.0, |az| (s - az.s_0) / az.s_norm)
    })
}
fn ikc_inv<const C: usize, const R: usize>(
    s: f64,
    constants: &KMatrixConstants<C, R>,
    channel: usize,
) -> SVector<Complex64, C> {
    let c_mat = c_matrix(s, constants);
    let i_mat = SMatrix::<Complex64, C, C>::identity();
    let k_mat = k_matrix(s, constants);
    let ikc_mat = i_mat + k_mat * c_mat;
    let ikc_inv_mat = ikc_mat.try_inverse().unwrap();
    ikc_inv_mat.row(channel).transpose()
}
fn p_vector<const C: usize, const R: usize>(
    betas: &SVector<Complex64, R>,
    pvector_constants: &SMatrix<Complex64, C, R>,
) -> SVector<Complex64, C> {
    SVector::<Complex64, C>::from_fn(|j, _| {
        (0..R).map(|a| betas[a] * pvector_constants[(j, a)]).sum()
    })
}

pub fn calculate_k_matrix<const C: usize, const R: usize, T>(
    _event: &T,
    betas: &SVector<Complex64, R>,
    precalc_data: &(SVector<Complex64, C>, SMatrix<Complex64, C, R>),
) -> Complex64
where
    T: Event + DaughterP4s,
{
    precalc_data.0.dot(&p_vector(betas, &precalc_data.1))
}

#[rustfmt::skip]
const F0: KMatrixConstants<5, 5> = KMatrixConstants {
    g: SMatrix::<f64, 5, 5>::new(
        0.74987, -0.01257, 0.02736, -0.15102, 0.36103,
        0.06401, 0.00204, 0.77413, 0.50999, 0.13112,
        -0.23417, -0.01032, 0.72283, 0.11934, 0.36792,
        0.0157, 0.267, 0.09214, 0.02742, -0.04025,
        -0.14242, 0.2278, 0.15981, 0.16272, -0.17397,
    ),
    c: SMatrix::<f64, 5, 5>::new(
        0.03728, 0.00000, -0.01398, -0.02203, 0.01397,
        0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
        -0.01398, 0.00000, 0.02349, 0.03101, -0.04003,
        -0.02203, 0.00000, 0.03101, -0.13769, -0.06722,
        0.01397, 0.00000, -0.04003, -0.06722, -0.28401,
    ),
    m1s: [0.13498, 0.26995, 0.49368, 0.54786, 0.54786],
    m2s: [0.13498, 0.26995, 0.49761, 0.54786, 0.95778],
    mrs: [0.51461, 0.90630, 1.23089, 1.46104, 1.69611],
    adler_zero: Some(AdlerZero {
        s_0: 0.0091125,
        s_norm: 1.0,
    }),
    l: 0,
};

#[derive(Default)]
pub struct KMatrixF0 {
    data: HashMap<Uuid, Vec<(SVector<Complex64, 5>, SMatrix<Complex64, 5, 5>)>>,
}

impl<T: Event + DaughterP4s + Sync> Amplitude<T> for KMatrixF0 {
    fn calculate(
        &mut self,
        parameters: &ParameterSet,
        dataset: &Dataset<T>,
    ) -> Vec<Complex64> {
        let precalcs = self.data.entry(dataset.uuid).or_insert_with(|| {
            dataset
                .events
                .par_iter()
                .map(|event| {
                    let s = (event.daughter_p4s()[0] + event.daughter_p4s()[1]).m2();
                    let barrier_mat = barrier_matrix(s, &F0);
                    let pvector_constants = SMatrix::<Complex64, 5, 5>::from_fn(|i, a| {
                        barrier_mat[(i, a)] * F0.g[(i, a)] / (F0.mrs[a].powi(2) - s)
                    });
                    (ikc_inv(s, &F0, 2), pvector_constants)
                })
                .collect()
        });
        let betas = SVector::<Complex64, 5>::new(
            Complex64::new(parameters.get("f0_500 re"),parameters.get("f0_500 im")),
            Complex64::new(
                parameters.get("f0_980 re"),
                parameters.get("f0_980 im"),
            ),
            Complex64::new(
                parameters.get("f0_1370 re"),
                parameters.get("f0_1370 im"),
            ),
            Complex64::new(
                parameters.get("f0_1500 re"),
                parameters.get("f0_1500 im"),
            ),
            Complex64::new(
                parameters.get("f0_1710 re"),
                parameters.get("f0_1710 im"),
            ),
        );
        dataset
            .events
            .par_iter()
            .zip(precalcs)
            .map(|(event, precalc)| calculate_k_matrix(event, &betas, precalc))
            .collect()
    }
}

#[rustfmt::skip]
const F2: KMatrixConstants<4, 4> = KMatrixConstants {
    g: SMatrix::<f64, 4, 4>::new(
        0.40033, 0.15479, -0.089, -0.00113,
        0.0182, 0.173, 0.32393, 0.15256,
        -0.06709, 0.22941, -0.43133, 0.23721,
        -0.49924, 0.19295, 0.27975, -0.03987,
    ),
    c: SMatrix::<f64, 4, 4>::new(
        -0.04319, 0.00000, 0.00984, 0.01028,
        0.00000, 0.00000, 0.00000, 0.00000,
        0.00984, 0.00000, -0.07344, 0.05533,
        0.01028, 0.00000, 0.05533, -0.05183,
    ),
    m1s: [0.13498, 0.26995, 0.49368, 0.54786],
    m2s: [0.13498, 0.26995, 0.49761, 0.54786],
    mrs: [1.15299, 1.48359, 1.72923, 1.96700],
    adler_zero: None,
    l: 2,
};

#[derive(Default)]
pub struct KMatrixF2 {
    data: HashMap<Uuid, Vec<(SVector<Complex64, 4>, SMatrix<Complex64, 4, 4>)>>,
}

impl<T: Event + DaughterP4s + Sync> Amplitude<T> for KMatrixF2 {
    fn calculate(
        &mut self,
        parameters: &ParameterSet,
        dataset: &Dataset<T>,
    ) -> Vec<Complex64> {
        let precalcs = self.data.entry(dataset.uuid).or_insert_with(|| {
            dataset
                .events
                .par_iter()
                .map(|event| {
                    let s = (event.daughter_p4s()[0] + event.daughter_p4s()[1]).m2();
                    let barrier_mat = barrier_matrix(s, &F2);
                    let pvector_constants = SMatrix::<Complex64, 4, 4>::from_fn(|i, a| {
                        barrier_mat[(i, a)] * F2.g[(i, a)] / (F2.mrs[a].powi(2) - s)
                    });
                    (ikc_inv(s, &F2, 2), pvector_constants)
                })
                .collect()
        });
        let betas = SVector::<Complex64, 4>::new(
            Complex64::new(
                parameters.get("f2_1270 re"),
                parameters.get("f2_1270 im"),
            ),
            Complex64::new(
                parameters.get("f2_1525 re"),
                parameters.get("f2_1525 im"),
            ),
            Complex64::new(
                parameters.get("f2_1810 re"),
                parameters.get("f2_1810 im"),
            ),
            Complex64::new(
                parameters.get("f2_1950 re"),
                parameters.get("f2_1950 im"),
            ),
        );
        dataset
            .events
            .par_iter()
            .zip(precalcs)
            .map(|(event, precalc)| calculate_k_matrix(event, &betas, precalc))
            .collect()
    }
}

#[rustfmt::skip]
const A0: KMatrixConstants<2, 2> = KMatrixConstants {
    g: SMatrix::<f64, 2, 2>::new(
        0.43215, -0.28825,
        0.19, 0.43372
    ),
    c: SMatrix::<f64, 2, 2>::new(
        0.00000, 0.00000, 
        0.00000, 0.00000
    ),
    m1s: [0.13498, 0.49368],
    m2s: [0.54786, 0.49761],
    mrs: [0.95395, 1.26767],
    adler_zero: None,
    l: 0,
};

#[derive(Default)]
pub struct KMatrixA0 {
    data: HashMap<Uuid, Vec<(SVector<Complex64, 2>, SMatrix<Complex64, 2, 2>)>>,
}

impl<T: Event + DaughterP4s + Sync> Amplitude<T> for KMatrixA0 {
    fn calculate(
        &mut self,
        parameters: &ParameterSet,
        dataset: &Dataset<T>,
    ) -> Vec<Complex64> {
        let precalcs = self.data.entry(dataset.uuid).or_insert_with(|| {
            dataset
                .events
                .par_iter()
                .map(|event| {
                    let s = (event.daughter_p4s()[0] + event.daughter_p4s()[1]).m2();
                    let barrier_mat = barrier_matrix(s, &A0);
                    let pvector_constants = SMatrix::<Complex64, 2, 2>::from_fn(|i, a| {
                        barrier_mat[(i, a)] * A0.g[(i, a)] / (A0.mrs[a].powi(2) - s)
                    });
                    (ikc_inv(s, &A0, 1), pvector_constants)
                })
                .collect()
        });
        let betas = SVector::<Complex64, 2>::new(
            Complex64::new(
                parameters.get("a0_980 re"),
                parameters.get("a0_980 im"),
            ),
            Complex64::new(
                parameters.get("a0_1450 re"),
                parameters.get("a0_1450 im"),
            ),
        );
        dataset
            .events
            .par_iter()
            .zip(precalcs)
            .map(|(event, precalc)| calculate_k_matrix(event, &betas, precalc))
            .collect()
    }
}

#[rustfmt::skip]
const A2: KMatrixConstants<3, 2> = KMatrixConstants {
    g: SMatrix::<f64, 3, 2>::new(
        0.30073, 0.21426, -0.09162,
        0.68567, 0.12543, 0.00184),
    c: SMatrix::<f64, 3, 3>::new(
        -0.40184, 0.00033, -0.08707,
        0.00033, -0.21416, -0.06193,
        -0.08707, -0.06193, -0.17435,
    ),
    m1s: [0.13498, 0.49368, 0.13498],
    m2s: [0.54786, 0.49761, 0.95778],
    mrs: [1.30080, 1.75351],
    adler_zero: None,
    l: 2,
};

#[derive(Default)]
pub struct KMatrixA2 {
    data: HashMap<Uuid, Vec<(SVector<Complex64, 3>, SMatrix<Complex64, 3, 2>)>>,
}

impl<T: Event + DaughterP4s + Sync> Amplitude<T> for KMatrixA2 {
    fn calculate(
        &mut self,
        parameters: &ParameterSet,
        dataset: &Dataset<T>,
    ) -> Vec<Complex64> {
        let precalcs = self.data.entry(dataset.uuid).or_insert_with(|| {
            dataset
                .events
                .par_iter()
                .map(|event| {
                    let s = (event.daughter_p4s()[0] + event.daughter_p4s()[1]).m2();
                    let barrier_mat = barrier_matrix(s, &A2);
                    let pvector_constants = SMatrix::<Complex64, 3, 2>::from_fn(|i, a| {
                        barrier_mat[(i, a)] * A2.g[(i, a)] / (A2.mrs[a].powi(2) - s)
                    });
                    (ikc_inv(s, &A2, 1), pvector_constants)
                })
                .collect()
        });
        let betas = SVector::<Complex64, 2>::new(
            Complex64::new(
                parameters.get("a2_1320 re"),
                parameters.get("a2_1320 im"),
            ),
            Complex64::new(
                parameters.get("a2_1700 re"),
                parameters.get("a2_1700 im"),
            ),
        );
        dataset
            .events
            .par_iter()
            .zip(precalcs)
            .map(|(event, precalc)| calculate_k_matrix(event, &betas, precalc))
            .collect()
    }
}
