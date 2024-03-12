use std::{f64::consts::PI, fmt::Display, fs::File};

use crate::{
    dataset::{
        extract_scalar, extract_vector, scalars_to_momentum, vectors_to_momenta, Dataset, Event,
        ReadType,
    },
    prelude::FourMomentum,
};

use nalgebra::{ComplexField, SMatrix, SVector, Vector3};
use num_complex::Complex64;
use polars::prelude::*;
use rayon::prelude::*;
use sphrs::{ComplexSH, Coordinates, SHEval};

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

#[derive(Debug)]
pub struct GlueXEvent {
    weight: f64,
    beam_p4: FourMomentum,
    recoil_p4: FourMomentum,
    daughter_p4s: Vec<FourMomentum>,
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
        writeln!(f, "Recoil P4: {}", self.beam_p4)?;
        writeln!(f, "Daughters:")?;
        for (i, p4) in self.daughter_p4s.iter().enumerate() {
            writeln!(f, "\t{i} -> {p4}")?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct GlueXPolarizedEvent {
    weight: f64,
    beam_p4: FourMomentum,
    recoil_p4: FourMomentum,
    daughter_p4s: Vec<FourMomentum>,
    eps: Vector3<f64>,
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
        writeln!(f, "Recoil P4: {}", self.beam_p4)?;
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

// NOTE TO SELF: for now we'll assume these are COM boosted, but we can add in a thing to do it
// automatically later, maybe a COM boost function
pub fn open_gluex(path: &str) -> Dataset<GlueXEvent> {
    let file = File::open(path).expect("Open Error");
    let dataframe = ParquetReader::new(file).finish().expect("Read Error");
    let e_beam = extract_scalar("E_Beam", &dataframe, ReadType::F32);
    let px_beam = extract_scalar("Px_Beam", &dataframe, ReadType::F32);
    let py_beam = extract_scalar("Py_Beam", &dataframe, ReadType::F32);
    let pz_beam = extract_scalar("Pz_Beam", &dataframe, ReadType::F32);
    let e_finalstate = extract_vector("E_FinalState", &dataframe, ReadType::F32);
    let px_finalstate = extract_vector("Px_FinalState", &dataframe, ReadType::F32);
    let py_finalstate = extract_vector("Py_FinalState", &dataframe, ReadType::F32);
    let pz_finalstate = extract_vector("Pz_FinalState", &dataframe, ReadType::F32);
    let beam_four_momentum = scalars_to_momentum(e_beam, px_beam, py_beam, pz_beam);
    let final_state_four_momentum =
        vectors_to_momenta(e_finalstate, px_finalstate, py_finalstate, pz_finalstate);
    let weight = extract_scalar("Weight", &dataframe, ReadType::F32);
    let mut dataset = Dataset::new();
    dataset.events = (beam_four_momentum, final_state_four_momentum, weight)
        .into_par_iter()
        .map(|(beam_p4, fs, weight)| GlueXEvent {
            weight,
            beam_p4,
            recoil_p4: fs[0],
            daughter_p4s: fs[1..].to_vec(),
        })
        .collect();
    dataset
}

pub fn open_gluex_polarized(path: &str) -> Dataset<GlueXPolarizedEvent> {
    let file = File::open(path).expect("Open Error");
    let dataframe = ParquetReader::new(file).finish().expect("Read Error");
    let e_beam = extract_scalar("E_Beam", &dataframe, ReadType::F32);
    let px_beam = extract_scalar("Px_Beam", &dataframe, ReadType::F32);
    let py_beam = extract_scalar("Py_Beam", &dataframe, ReadType::F32);
    let e_finalstate = extract_vector("E_FinalState", &dataframe, ReadType::F32);
    let px_finalstate = extract_vector("Px_FinalState", &dataframe, ReadType::F32);
    let py_finalstate = extract_vector("Py_FinalState", &dataframe, ReadType::F32);
    let pz_finalstate = extract_vector("Pz_FinalState", &dataframe, ReadType::F32);
    let zero_vec = vec![0.0; e_beam.len()];
    let beam_four_momentum =
        scalars_to_momentum(e_beam.clone(), zero_vec.clone(), zero_vec, e_beam);
    let eps_array: Vec<Vector3<f64>> = px_beam
        .into_par_iter()
        .zip(py_beam.into_par_iter())
        .map(|(px, py)| Vector3::new(px, py, 0.0))
        .collect();
    let final_state_four_momentum =
        vectors_to_momenta(e_finalstate, px_finalstate, py_finalstate, pz_finalstate);
    let weight = extract_scalar("Weight", &dataframe, ReadType::F32);
    let mut dataset = Dataset::new();
    dataset.events = (
        beam_four_momentum,
        final_state_four_momentum,
        weight,
        eps_array,
    )
        .into_par_iter()
        .map(|(beam_p4, fs, weight, eps)| GlueXPolarizedEvent {
            weight,
            beam_p4,
            recoil_p4: fs[0],
            daughter_p4s: fs[1..].to_vec(),
            eps,
        })
        .collect();
    dataset
}

#[derive(Clone, Copy)]
#[rustfmt::skip]
pub enum Wave {
    S0, S,
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

pub fn ylm<T>(event: &T, wave: Wave) -> Complex64
where
    T: Event + BeamP4 + RecoilP4 + DaughterP4s,
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
    ComplexSH::Spherical.eval(wave.l(), wave.m(), &p)
}

pub enum Reflectivity {
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

fn chi_plus(s: f64, m1: f64, m2: f64) -> Complex64 {
    (1.0 - ((m1 + m2) * (m1 + m2)) / s).into()
}

fn chi_minus(s: f64, m1: f64, m2: f64) -> Complex64 {
    (1.0 - ((m1 - m2) * (m1 - m2)) / s).into()
}

fn rho(s: f64, m1: f64, m2: f64) -> Complex64 {
    (chi_plus(s, m1, m2) * chi_minus(s, m1, m2)).sqrt()
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
    let q = rho(s, m1, m2) * s.sqrt() / 2.0;
    q * q / (0.1973 * 0.1973)
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
            .into_par_iter()
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
    betas: &SVector<Complex64, C>,
    s: f64,
    constants: &KMatrixConstants<C, R>,
    barrier_factor: &SMatrix<Complex64, C, R>,
) -> SVector<Complex64, C> {
    SVector::<Complex64, C>::from_fn(|j, _| {
        (0..R)
            .into_par_iter()
            .map(|a| {
                betas[a] * constants.g[(j, a)] / (constants.mrs[a].powi(2) - s)
                    * barrier_factor[(j, a)]
            })
            .sum()
    })
}
fn calculate_k_matrix<const C: usize, const R: usize>(
    betas: &SVector<Complex64, C>,
    s: f64,
    constants: &KMatrixConstants<C, R>,
    barrier_factor: &SMatrix<Complex64, C, R>,
    ikc_inv_vec: &SVector<Complex64, C>,
) -> Complex64 {
    ikc_inv_vec.dot(&p_vector(betas, s, constants, barrier_factor))
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

pub fn precalculate_k_matrix_f0<T>(
    event: &T,
    channel: usize,
) -> (SMatrix<Complex64, 5, 5>, SVector<Complex64, 5>)
where
    T: Event + DaughterP4s,
{
    let s = (event.daughter_p4s()[0] + event.daughter_p4s()[1]).m2();
    (barrier_matrix(s, &F0), ikc_inv(s, &F0, channel))
}

pub fn calculate_k_matrix_f0<T>(
    event: &T,
    betas: &SVector<Complex64, 5>,
    barrier_factor: &SMatrix<Complex64, 5, 5>,
    ikc_inv_vec: &SVector<Complex64, 5>,
) -> Complex64
where
    T: Event + DaughterP4s,
{
    let s = (event.daughter_p4s()[0] + event.daughter_p4s()[1]).m2();
    calculate_k_matrix(betas, s, &F0, barrier_factor, ikc_inv_vec)
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
