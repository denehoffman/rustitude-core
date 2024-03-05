#![allow(unused_imports)]
use rustc_hash::FxHashMap as HashMap;
use std::{f64::consts::PI, fmt::Display};
use uuid::Uuid;

use anyinput::anyinput;
use nalgebra::{ComplexField, Vector3};
use ndarray::{array, linalg::Dot, Array1, Array2, Array3, Axis};
use ndarray_linalg::Inverse;
use num_complex::Complex64;
use rayon::prelude::*;
use sphrs::{ComplexSH, Coordinates, SHEval};
use std::iter::repeat;

use crate::prelude::*;

/// Open a `GlueX` ROOT data file (flat tree) by `path`. `polarized` is a flag which is `true` if the
/// data file has polarization information included in the `"Px_Beam"` and `"Py_Beam"` branches.
///
/// # Panics
///
/// Panics if it can't find any of the required branches, or if they contain data with an
/// unexpected type.
///
/// # Errors
///
/// Will raise [`PolarsError`] in the event that any of the branches aren't read or converted
/// properly.
pub fn open_gluex(path: &str, polarized: bool) -> Result<Dataset, DatasetError> {
    let dataframe = open_parquet(path).expect("Read error");
    let mut dataset = Dataset::from_size(dataframe.height());
    let e_beam = extract_scalar("E_Beam", &dataframe, ReadType::F32);
    let px_beam = extract_scalar("Px_Beam", &dataframe, ReadType::F32);
    let py_beam = extract_scalar("Py_Beam", &dataframe, ReadType::F32);
    let pz_beam = extract_scalar("Pz_Beam", &dataframe, ReadType::F32);
    if polarized {
        let zero_vec = vec![0.0; e_beam.len()];
        let beam_p4 = scalars_to_momentum_par(e_beam.clone(), zero_vec.clone(), zero_vec, e_beam);
        let eps = px_beam
            .into_par_iter()
            .zip(py_beam.into_par_iter())
            .map(|(px, py)| array![px, py, 0.0])
            .collect();
        dataset.insert_vector("Beam P4", beam_p4)?;
        dataset.insert_vector("EPS", eps)?;
    } else {
        let beam_p4 = scalars_to_momentum_par(e_beam, px_beam, py_beam, pz_beam);
        dataset.insert_vector("Beam P4", beam_p4)?;
    }
    let weight = extract_scalar("Weight", &dataframe, ReadType::F32);
    dataset.insert_scalar("Weight", weight)?;
    let e_finalstate = extract_vector("E_FinalState", &dataframe, ReadType::F32);
    let px_finalstate = extract_vector("Px_FinalState", &dataframe, ReadType::F32);
    let py_finalstate = extract_vector("Py_FinalState", &dataframe, ReadType::F32);
    let pz_finalstate = extract_vector("Pz_FinalState", &dataframe, ReadType::F32);
    let final_state_p4 =
        vectors_to_momenta_par(e_finalstate, px_finalstate, py_finalstate, pz_finalstate);
    dataset.insert_vector("Recoil P4", final_state_p4[0].clone())?;
    dataset.insert_vector("Decay1 P4", final_state_p4[1].clone())?;
    dataset.insert_vector("Decay2 P4", final_state_p4[2].clone())?;
    Ok(dataset)
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

#[derive(Clone)]
pub struct Ylm {
    pub wave: Wave,
    ylm: HashMap<Uuid, Vec<Complex64>>,
}

impl Display for Ylm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Ylm {}", self.wave)
    }
}

impl Ylm {
    pub fn new(wave: Wave) -> Self {
        Ylm {
            wave,
            ylm: HashMap::default(),
        }
    }
    fn calculate_ylm(&mut self, dataset: &Dataset) -> Vec<Complex64> {
        let beam_p4 = dataset.vector("Beam P4").unwrap();
        let recoil_p4 = dataset.vector("Recoil P4").unwrap();
        let decay1_p4 = dataset.vector("Decay1 P4").unwrap();
        let decay2_p4 = dataset.vector("Decay2 P4").unwrap();

        (beam_p4, recoil_p4, decay1_p4, decay2_p4)
            .into_par_iter()
            .map(|(beam_arr, recoil_arr, decay1_arr, decay2_arr)| {
                let beam_lab = FourMomentum::from(beam_arr);
                let proton_lab = FourMomentum::from(recoil_arr);
                let decay1_lab = FourMomentum::from(decay1_arr);
                let decay2_lab = FourMomentum::from(decay2_arr);
                let final_state_lab = decay1_lab + decay2_lab + proton_lab;
                let resonance_lab = decay1_lab + decay2_lab;

                let beam = beam_lab.boost_along(&final_state_lab);
                let recoil = proton_lab.boost_along(&final_state_lab);
                let resonance = resonance_lab.boost_along(&final_state_lab);
                let decay1 = decay1_lab.boost_along(&final_state_lab);

                let recoil_res = recoil.boost_along(&resonance);
                let decay1_res = decay1.boost_along(&resonance);

                let z = -1.0 * recoil_res.momentum().normalize();
                let y = beam.momentum().cross(&(-1.0 * recoil.momentum()));
                let x = y.cross(&z);
                let decay1_p3 = decay1_res.momentum();

                let p =
                    Coordinates::cartesian(decay1_p3.dot(&x), decay1_p3.dot(&y), decay1_p3.dot(&z));
                ComplexSH::Spherical.eval(self.wave.l(), self.wave.m(), &p)
            })
            .collect()
    }
}

impl Amplitude for Ylm {
    fn evaluate(
        &mut self,
        dataset: &Dataset,
        _parameters: &HashMap<String, f64>,
    ) -> Vec<Complex64> {
        if let Some(res) = self.ylm.get(&dataset.get_uuid()) {
            res.to_vec()
        } else {
            let res = self.calculate_ylm(dataset);
            self.ylm.insert(dataset.get_uuid(), res);
            self.ylm.get(&dataset.get_uuid()).unwrap().to_vec()
        }
    }
}

pub enum Reflectivity {
    Positive,
    Negative,
}

pub enum Part {
    Real,
    Imag,
}

pub struct Zlm {
    pub wave: Wave,
    pub reflectivity: Reflectivity,
    pub part: Part,
    zlm: HashMap<Uuid, Vec<Complex64>>,
}

impl Display for Zlm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let part = match self.part {
            Part::Real => "Re",
            Part::Imag => "Im",
        };
        let refl = match self.reflectivity {
            Reflectivity::Positive => "+",
            Reflectivity::Negative => "-",
        };
        write!(f, "{}[Zlm {} ({})]", part, self.wave, refl)
    }
}

impl Zlm {
    pub fn new(wave: Wave, reflectivity: Reflectivity, part: Part) -> Self {
        Zlm {
            wave,
            reflectivity,
            part,
            zlm: HashMap::default(),
        }
    }
    fn calculate_zlm(&mut self, dataset: &Dataset) -> Vec<Complex64> {
        let beam_p4 = dataset.vector("Beam P4").unwrap();
        let recoil_p4 = dataset.vector("Recoil P4").unwrap();
        let decay1_p4 = dataset.vector("Decay1 P4").unwrap();
        let decay2_p4 = dataset.vector("Decay2 P4").unwrap();
        let eps_vec = dataset.vector("EPS").unwrap();

        let sign = match self.reflectivity {
            Reflectivity::Positive => 1.0,
            Reflectivity::Negative => -1.0,
        } * match self.part {
            Part::Real => 1.0,
            Part::Imag => -1.0,
        };
        (beam_p4, recoil_p4, decay1_p4, decay2_p4, eps_vec)
            .into_par_iter()
            .map(|(beam_arr, recoil_arr, decay1_arr, decay2_arr, eps_arr)| {
                let beam_lab = FourMomentum::from(beam_arr);
                let proton_lab = FourMomentum::from(recoil_arr);
                let decay1_lab = FourMomentum::from(decay1_arr);
                let decay2_lab = FourMomentum::from(decay2_arr);
                let final_state_lab = decay1_lab + decay2_lab + proton_lab;
                let resonance_lab = decay1_lab + decay2_lab;

                let beam = beam_lab.boost_along(&final_state_lab);
                let recoil = proton_lab.boost_along(&final_state_lab);
                let resonance = resonance_lab.boost_along(&final_state_lab);
                let decay1 = decay1_lab.boost_along(&final_state_lab);

                let recoil_res = recoil.boost_along(&resonance);
                let decay1_res = decay1.boost_along(&resonance);

                let z = -1.0 * recoil_res.momentum().normalize();
                let y = beam.momentum().cross(&(-1.0 * recoil.momentum()));
                let x = y.cross(&z);
                let decay1_p3 = decay1_res.momentum();

                let p =
                    Coordinates::cartesian(decay1_p3.dot(&x), decay1_p3.dot(&y), decay1_p3.dot(&z));
                let ylm = ComplexSH::Spherical.eval(self.wave.l(), self.wave.m(), &p);
                let eps = Vector3::from_vec(eps_arr.to_vec());
                let big_phi = y
                    .dot(&eps)
                    .atan2(beam.momentum().normalize().dot(&eps.cross(&y)));
                let pgamma = eps.norm();

                let phase = Complex64::cis(-big_phi);
                let zlm = ylm * phase;
                Complex64::new((1.0 + sign * pgamma).sqrt() * zlm.re, 0.0)
            })
            .collect()
    }
}
impl Amplitude for Zlm {
    fn evaluate(
        &mut self,
        dataset: &Dataset,
        _parameters: &HashMap<String, f64>,
    ) -> Vec<Complex64> {
        if let Some(res) = self.zlm.get(&dataset.get_uuid()) {
            res.to_vec()
        } else {
            let res = self.calculate_zlm(dataset);
            self.zlm.insert(dataset.get_uuid(), res);
            self.zlm.get(&dataset.get_uuid()).unwrap().to_vec()
        }
    }
}

#[derive(Clone)]
pub struct KMatrixConstants {
    pub g: Array2<f64>,
    pub m: Array1<f64>,
    pub c: Array2<f64>,
    pub m1: Array1<f64>,
    pub m2: Array1<f64>,
    pub n_resonances: usize,
    pub n_channels: usize,
    pub wave: Wave,
}

impl KMatrixConstants {
    fn chi_plus(&self, s: f64, channel: usize) -> Complex64 {
        (1.0 - ((&self.m1 + &self.m2) * (&self.m1 + &self.m2))[channel] / s).into()
    }

    fn chi_minus(&self, s: f64, channel: usize) -> Complex64 {
        (1.0 - ((&self.m1 - &self.m2) * (&self.m1 - &self.m2))[channel] / s).into()
    }

    fn rho(&self, s: f64, channel: usize) -> Complex64 {
        (self.chi_plus(s, channel) * self.chi_minus(s, channel)).sqrt()
    }
    fn c_matrix(&self, s: f64) -> Array2<Complex64> {
        Array2::from_diag(&Array1::from_shape_fn(self.n_channels, |channel| {
            self.rho(s, channel) / PI
                * ((self.chi_plus(s, channel) + self.rho(s, channel))
                    / (self.chi_plus(s, channel) - self.rho(s, channel)))
                .ln()
                + self.chi_plus(s, channel) / PI
                    * ((&self.m2 - &self.m1) / (&self.m1 + &self.m2))[channel]
                    * Complex64::from((&self.m2 / &self.m1)[channel]).ln()
        }))
    }
    fn z(&self, s: f64, channel: usize) -> Complex64 {
        let q = self.rho(s, channel) * Complex64::sqrt(s.into()) / 2.0;
        q * q / (0.1973 * 0.1973)
    }
    fn blatt_weisskopf(&self, s: f64, channel: usize) -> Complex64 {
        let z = self.z(s, channel);
        match self.wave.l() {
            0 => 1.0.into(),
            1 => ((2.0 * z) / (z + 1.0)).sqrt(),
            2 => ((13.0 * z.powi(2)) / ((z - 3.0).powi(2) + 9.0 * z)).sqrt(),
            3 => ((277.0 * z.powi(3)) / (z * (z - 15.0).powi(2) + 9.0 * (2.0 * z - 5.0).powi(2)))
                .sqrt(),
            4 => ((12746.0 * z.powi(4)) / (z.powi(2) - 45.0 * z + 105.0).powi(2)
                + 25.0 * z * (2.0 * z - 21.0).powi(2))
            .sqrt(),
            l => panic!("L = {l} is not yet implemented"),
        }
    }
    fn barrier_factor(&self, s: f64, channel: usize, resonance: usize) -> Complex64 {
        let numerator = self.blatt_weisskopf(s, channel);
        let denominator = self.blatt_weisskopf(self.m[resonance].powi(2), channel);
        numerator / denominator
    }
}

#[derive(Clone)]
pub struct AdlerZero {
    pub s_0: f64,
    pub s_norm: f64,
}

pub struct FrozenKMatrix {
    selected_channel: usize,
    constants: KMatrixConstants,
    barrier_factor: HashMap<Uuid, Vec<CMatrix64>>,
    ikc_vector: HashMap<Uuid, Vec<CVector64>>,
    adler_zero: Option<AdlerZero>,
}

impl Display for FrozenKMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Kmatrix placeholder")
    }
}

impl FrozenKMatrix {
    pub fn new(
        selected_channel: usize,
        constants: &KMatrixConstants,
        adler_zero: Option<&AdlerZero>,
    ) -> Self {
        FrozenKMatrix {
            selected_channel,
            constants: constants.clone(),
            barrier_factor: HashMap::default(),
            ikc_vector: HashMap::default(),
            adler_zero: adler_zero.cloned(),
        }
    }
    fn calculate_barrier_factor(&self, dataset: &Dataset) -> Vec<CMatrix64> {
        let decay1_p4 = dataset.vector("Decay1 P4").unwrap();
        let decay2_p4 = dataset.vector("Decay2 P4").unwrap();
        (decay1_p4, decay2_p4)
            .into_par_iter()
            .map(|(decay1_arr, decay2_arr)| {
                let decay1_lab = FourMomentum::from(decay1_arr);
                let decay2_lab = FourMomentum::from(decay2_arr);
                let s = (decay1_lab + decay2_lab).m2();
                Array2::from_shape_fn(
                    (self.constants.n_channels, self.constants.n_resonances),
                    |(i, a)| self.constants.barrier_factor(s, i, a),
                )
            })
            .collect()
    }
    fn calculate_k_matrix(
        &mut self,
        dataset: &Dataset,
        barrier_factor: &Vec<CMatrix64>,
    ) -> Vec<CVector64> {
        let decay1_p4 = dataset.vector("Decay1 P4").unwrap();
        let decay2_p4 = dataset.vector("Decay2 P4").unwrap();
        (decay1_p4, decay2_p4, barrier_factor)
            .into_par_iter()
            .map(|(decay1_arr, decay2_arr, bf)| {
                let decay1_lab = FourMomentum::from(decay1_arr);
                let decay2_lab = FourMomentum::from(decay2_arr);
                let s = (decay1_lab + decay2_lab).m2();
                let k_ija = Array3::from_shape_fn(
                    (
                        self.constants.n_channels,
                        self.constants.n_channels,
                        self.constants.n_resonances,
                    ),
                    |(i, j, a)| {
                        bf[[i, a]]
                            * ((self.constants.g[[i, a]] * self.constants.g[[j, a]])
                                / (self.constants.m[a].powi(2) - s)
                                + self.constants.c[[i, j]])
                            * bf[[j, a]]
                    },
                );
                let k_mat = match self.adler_zero {
                    Some(AdlerZero { s_0, s_norm }) => {
                        Complex64::from((s - s_0) / s_norm) * k_ija.sum_axis(Axis(2))
                    }
                    None => k_ija.sum_axis(Axis(2)),
                };
                let c_mat = self.constants.c_matrix(s);
                let i_mat: Array2<Complex64> = Array2::eye(self.constants.n_channels);
                let ikc_mat = i_mat + k_mat * c_mat;
                let ikc_mat_inv = ikc_mat.inv().unwrap();
                ikc_mat_inv.row(self.selected_channel).to_owned()
            })
            .collect()
    }
}

impl Amplitude for FrozenKMatrix {
    fn parameter_names(&self) -> Option<Vec<String>> {
        Some(
            (0..self.constants.n_resonances)
                .flat_map(|i| vec![format!("Resonance {i} re"), format!("Resonance {i} im")])
                .collect(),
        )
    }
    fn evaluate(&mut self, dataset: &Dataset, parameters: &HashMap<String, f64>) -> Vec<Complex64> {
        let barrier_factor = if let Some(bf) = self.barrier_factor.get(&dataset.get_uuid()) {
            bf.to_vec()
        } else {
            let res = self.calculate_barrier_factor(dataset);
            self.barrier_factor.insert(dataset.get_uuid(), res);
            self.barrier_factor
                .get(&dataset.get_uuid())
                .unwrap()
                .to_vec()
        };
        let ikc_vector = if let Some(ikc_vec) = self.ikc_vector.get(&dataset.get_uuid()) {
            ikc_vec.to_vec()
        } else {
            let res = self.calculate_k_matrix(dataset, &barrier_factor);
            self.ikc_vector.insert(dataset.get_uuid(), res);
            self.ikc_vector.get(&dataset.get_uuid()).unwrap().to_vec()
        };
        let betas = Array1::from_shape_fn(self.constants.n_resonances, |i| {
            Complex64::new(
                *parameters.get(&format!("Resonance {i} re")).unwrap(),
                *parameters.get(&format!("Resonance {i} im")).unwrap(),
            )
        });

        let decay1_p4 = dataset.vector("Decay1 P4").unwrap();
        let decay2_p4 = dataset.vector("Decay2 P4").unwrap();
        (decay1_p4, decay2_p4, ikc_vector, barrier_factor)
            .into_par_iter()
            .map(|(decay1_arr, decay2_arr, ikc, bf)| {
                let decay1_lab = FourMomentum::from(decay1_arr);
                let decay2_lab = FourMomentum::from(decay2_arr);
                let s = (decay1_lab + decay2_lab).m2();

                let p_ja = Array2::from_shape_fn(
                    (self.constants.n_channels, self.constants.n_resonances),
                    |(j, a)| {
                        ((betas[a] * self.constants.g[[j, a]]) / (self.constants.m[a].powi(2) - s))
                            * bf[[j, a]]
                    },
                );
                let p_vec = p_ja.sum_axis(Axis(1));
                ikc.dot(&p_vec)
            })
            .collect()
    }
}
