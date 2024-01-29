#![allow(unused_imports)]
use std::{collections::HashMap, f64::consts::PI};

use anyinput::anyinput;
use nalgebra::Vector3;
use ndarray::{array, linalg::Dot, Array1, Array2, Array3, Axis};
use ndarray_linalg::Inverse;
use num_complex::Complex64;
use rayon::prelude::*;
use sphrs::{ComplexSH, Coordinates, SHEval};
use uuid::Uuid;

use crate::{node::VectorNode, prelude::*};

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

pub struct ResonanceMass {
    pub mass: String,
}

impl ResonanceMass {
    pub fn new() -> Self {
        ResonanceMass::default()
    }
}

impl Default for ResonanceMass {
    fn default() -> Self {
        ResonanceMass {
            mass: "Resonance Mass".to_string(),
        }
    }
}
impl ResourceNode for ResonanceMass {
    fn eval(&self, ds: &mut Dataset) -> () {
        if ds.contains_scalar(&self.mass) {
            return;
        }
        let d1 = ds.vector("Decay1 P4").unwrap();
        let d2 = ds.vector("Decay2 P4").unwrap();
        let m = (d1, d2)
            .into_par_iter()
            .map(|(a, b)| {
                let a_p4 = FourMomentum::from(a);
                let b_p4 = FourMomentum::from(b);
                (a_p4 + b_p4).m()
            })
            .collect();
        ds.insert_scalar(&self.mass, m).unwrap();
    }
}

#[derive(Clone)]
pub struct HelicityVec {
    pub x: String,
    pub y: String,
    pub z: String,
    pub resonance: String,
}

impl HelicityVec {
    pub fn new() -> Self {
        HelicityVec::default()
    }
}

impl Default for HelicityVec {
    fn default() -> Self {
        HelicityVec {
            x: "Helicity X".to_string(),
            y: "Helicity Y".to_string(),
            z: "Helicity Z".to_string(),
            resonance: "Helicity Resonance Vec".to_string(),
        }
    }
}
impl ResourceNode for HelicityVec {
    fn eval(&self, ds: &mut Dataset) -> () {
        if ds.contains_vector(&self.x)
            && ds.contains_vector(&self.y)
            && ds.contains_vector(&self.z)
            && ds.contains_vector(&self.resonance)
        {
            return;
        }
        let beam_p4 = ds.vector("Beam P4").unwrap();
        let recoil_p4 = ds.vector("Recoil P4").unwrap();
        let decay1_p4 = ds.vector("Decay1 P4").unwrap();
        let decay2_p4 = ds.vector("Decay2 P4").unwrap();
        let (x, (y, (z, v))): (
            Vec<Array1<f64>>,
            (Vec<Array1<f64>>, (Vec<Array1<f64>>, Vec<Array1<f64>>)),
        ) = (beam_p4, recoil_p4, decay1_p4, decay2_p4)
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
                (
                    array![x.x, x.y, x.z],
                    (
                        array![y.x, y.y, y.z],
                        (
                            array![z.x, z.y, z.z],
                            array![decay1_p3.dot(&x), decay1_p3.dot(&y), decay1_p3.dot(&z)],
                        ),
                    ),
                )
            })
            .unzip();
        ds.insert_vector(&self.x, x).unwrap();
        ds.insert_vector(&self.y, y).unwrap();
        ds.insert_vector(&self.z, z).unwrap();
        ds.insert_vector(&self.resonance, v).unwrap();
    }
}

#[derive(Clone, Copy)]
#[rustfmt::skip]
pub enum Wave {
    S0,
    Pn1, P0, P1,
    Dn2, Dn1, D0, D1, D2,
    Fn3, Fn2, Fn1, F0, F1, F2, F3,
}

impl Wave {
    pub fn l(&self) -> i64 {
        match self {
            Self::S0 => 0,
            Self::Pn1 | Self::P0 | Self::P1 => 1,
            Self::Dn2 | Self::Dn1 | Self::D0 | Self::D1 | Self::D2 => 2,
            Self::Fn3 | Self::Fn2 | Self::Fn1 | Self::F0 | Self::F1 | Self::F2 | Self::F3 => 3,
        }
    }
    pub fn m(&self) -> i64 {
        match self {
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

impl ToString for Wave {
    fn to_string(&self) -> String {
        let l_string = match self {
            Self::S0 => "S",
            Self::Pn1 | Self::P0 | Self::P1 => "P",
            Self::Dn2 | Self::Dn1 | Self::D0 | Self::D1 | Self::D2 => "D",
            Self::Fn3 | Self::Fn2 | Self::Fn1 | Self::F0 | Self::F1 | Self::F2 | Self::F3 => "F",
        };
        format!("{} {:+}", l_string, self.m())
    }
}

#[derive(Clone)]
pub struct Ylm {
    pub wave: Wave,
    pub ylm: String,
    helicity_vec: HelicityVec,
}

impl Ylm {
    pub fn new(wave: Wave) -> Self {
        Ylm {
            wave,
            ylm: format!("Ylm ({})", wave.clone().to_string()),
            helicity_vec: HelicityVec::default(),
        }
    }
}

impl From<Wave> for Ylm {
    fn from(wave: Wave) -> Self {
        Ylm::new(wave)
    }
}

impl ResourceNode for Ylm {
    fn eval(&self, ds: &mut Dataset) -> () {
        if ds.contains_cscalar(&self.ylm) {
            return;
        }
        let ylm = ds
            .vector(&self.helicity_vec.resonance)
            .unwrap()
            .par_iter()
            .map(|xyz| {
                let p = Coordinates::cartesian(xyz[0], xyz[1], xyz[2]);
                ComplexSH::Spherical.eval(self.wave.l(), self.wave.m(), &p)
            })
            .collect();
        ds.insert_cscalar(&self.ylm, ylm).unwrap();
    }
    fn resources(&self) -> Vec<&dyn ResourceNode> {
        vec![&self.helicity_vec]
    }
}

#[derive(Clone)]
pub struct Polarization {
    pub big_phi: String,
    pub pgamma: String,
    helicity_vec: HelicityVec,
}

impl Polarization {
    pub fn new() -> Self {
        Polarization::default()
    }
}

impl Default for Polarization {
    fn default() -> Self {
        Polarization {
            big_phi: "Big Phi".to_string(),
            pgamma: "PGamma".to_string(),
            helicity_vec: HelicityVec::default(),
        }
    }
}

impl ResourceNode for Polarization {
    fn eval(&self, ds: &mut Dataset) -> () {
        if ds.contains_scalar(&self.big_phi) && ds.contains_scalar(&self.pgamma) {
            return;
        }
        let eps_vec = ds.vector("EPS").unwrap();
        let ys = ds.vector(&self.helicity_vec.y).unwrap();
        let beam_p4 = ds.vector("Beam P4").unwrap();
        let recoil_p4 = ds.vector("Recoil P4").unwrap();
        let decay1_p4 = ds.vector("Decay1 P4").unwrap();
        let decay2_p4 = ds.vector("Decay2 P4").unwrap();
        let (big_phi, pgamma): (Vec<f64>, Vec<f64>) =
            (eps_vec, ys, beam_p4, recoil_p4, decay1_p4, decay2_p4)
                .into_par_iter()
                .map(
                    |(eps_arr, y_arr, beam_arr, recoil_arr, decay1_arr, decay2_arr)| {
                        let beam_lab = FourMomentum::from(beam_arr);
                        let proton_lab = FourMomentum::from(recoil_arr);
                        let decay1_lab = FourMomentum::from(decay1_arr);
                        let decay2_lab = FourMomentum::from(decay2_arr);
                        let final_state_lab = decay1_lab + decay2_lab + proton_lab;
                        let beam = beam_lab.boost_along(&final_state_lab);
                        let eps = Vector3::from_vec(eps_arr.to_vec());
                        let y = Vector3::from_vec(y_arr.to_vec());

                        let bp = y
                            .dot(&eps)
                            .atan2(beam.momentum().normalize().dot(&eps.cross(&y)));
                        let pg = eps.norm();
                        (bp, pg)
                    },
                )
                .unzip();
        ds.insert_scalar(&self.big_phi, big_phi).unwrap();
        ds.insert_scalar(&self.pgamma, pgamma).unwrap();
    }
    fn resources(&self) -> Vec<&dyn ResourceNode> {
        vec![&self.helicity_vec]
    }
}

#[derive(Clone)]
pub enum Reflectivity {
    Positive,
    Negative,
}

impl ToString for Reflectivity {
    fn to_string(&self) -> String {
        match self {
            Self::Positive => "+".to_string(),
            Self::Negative => "-".to_string(),
        }
    }
}

#[derive(Clone)]
pub struct Zlm {
    pub zlm: String,
    reflectivity: Reflectivity,
    ylm: Ylm,
    polarization: Polarization,
}

impl Zlm {
    pub fn new(wave: Wave, reflectivity: Reflectivity) -> Self {
        Zlm {
            zlm: format!("Zlm ({})", wave.clone().to_string()),
            reflectivity,
            ylm: wave.into(),
            polarization: Polarization::default(),
        }
    }
}

impl ResourceNode for Zlm {
    fn eval(&self, ds: &mut Dataset) -> () {
        if ds.contains_cscalar(&self.zlm) {
            return;
        }
        let ylm_vec = ds.cscalar(&self.ylm.ylm).unwrap();
        let big_phi_vec = ds.scalar(&self.polarization.big_phi).unwrap();
        let pgamma_vec = ds.scalar(&self.polarization.pgamma).unwrap();
        let zlm = match self.reflectivity {
            Reflectivity::Positive => (ylm_vec, big_phi_vec, pgamma_vec)
                .into_par_iter()
                .map(|(ylm, big_phi, pgamma)| {
                    let phase = Complex64::cis(-big_phi);
                    let zlm = ylm * phase;
                    Complex64::new(
                        (1.0 + pgamma).sqrt() * zlm.re,
                        (1.0 - pgamma).sqrt() * zlm.im,
                    )
                })
                .collect(),
            Reflectivity::Negative => (ylm_vec, big_phi_vec, pgamma_vec)
                .into_par_iter()
                .map(|(ylm, big_phi, pgamma)| {
                    let phase = Complex64::cis(-big_phi);
                    let zlm = ylm * phase;
                    Complex64::new(
                        (1.0 - pgamma).sqrt() * zlm.re,
                        (1.0 + pgamma).sqrt() * zlm.im,
                    )
                })
                .collect(),
        };
        ds.insert_cscalar(&self.zlm, zlm).unwrap();
    }

    fn resources(&self) -> Vec<&dyn ResourceNode> {
        vec![&self.ylm, &self.polarization]
    }
}

// pub struct Zlm {
//     pub wave: Wave,
//     pub r: Reflectivity,
//     pub particle_info: ParticleInfo,
// }
//
// impl ToString for Zlm {
//     fn to_string(&self) -> String {
//         format!("Zlm ({}){}", self.wave.to_string(), self.r.to_string())
//     }
// }
//
// impl IntoVariable for Zlm {
//     fn into_variable(self) -> Variable {
//         Variable::CScalar(
//             CScalarVariableBuilder::default()
//                 .name(self.to_string())
//                 .function(move |entry: &Entry| {
//                     let beam_p4_lab = entry.momentum("Beam P4").unwrap();
//                     let fs_p4s_lab = entry.momenta("Final State P4").unwrap();
//                     let fs_p4_lab = &fs_p4s_lab.iter().sum();
//                     let recoil_p4_lab = &fs_p4s_lab[self.particle_info.recoil_index];
//                     let resonance_p4_lab: &FourMomentum = &self
//                         .particle_info
//                         .resonance_indices
//                         .iter()
//                         .map(|i| &fs_p4s_lab[*i])
//                         .sum();
//                     let daughter_p4_lab = &fs_p4s_lab[self.particle_info.daughter_index];
//
//                     let beam_p4 = &beam_p4_lab.boost_along(fs_p4_lab);
//                     let recoil_p4 = &recoil_p4_lab.boost_along(fs_p4_lab);
//                     let resonance_p4 = &resonance_p4_lab.boost_along(fs_p4_lab);
//                     let daughter_p4 = &daughter_p4_lab.boost_along(fs_p4_lab);
//
//                     let recoil_p4_res = &recoil_p4.boost_along(resonance_p4);
//                     let daughter_p4_res = &daughter_p4.boost_along(resonance_p4);
//
//                     let z = -1.0 * recoil_p4_res.momentum().normalize();
//                     let y = beam_p4
//                         .momentum()
//                         .cross(&(-1.0 * recoil_p4.momentum()))
//                         .normalize();
//                     let x = y.cross(&z);
//
//                     let daughter_p3_res = daughter_p4_res.momentum();
//
//                     let p = Coordinates::cartesian(
//                         daughter_p3_res.dot(&x),
//                         daughter_p3_res.dot(&y),
//                         daughter_p3_res.dot(&z),
//                     );
//                     let ylm = ComplexSH::Spherical.eval(self.wave.l(), self.wave.m(), &p);
//
//                     // Linear polarization
//                     let eps = Vector3::from_vec(entry.vector("EPS").unwrap().to_vec());
//                     let big_phi = y
//                         .dot(&eps)
//                         .atan2(beam_p4.momentum().normalize().dot(&eps.cross(&y)));
//                     let phase = Complex64::cis(-big_phi);
//                     let pgamma = eps.norm();
//
//                     let zlm = ylm * phase;
//
//                     match self.r {
//                         Reflectivity::Positive => Complex64 {
//                             re: (1.0 + pgamma).sqrt() * zlm.re,
//                             im: (1.0 - pgamma).sqrt() * zlm.im,
//                         },
//                         Reflectivity::Negative => Complex64 {
//                             re: (1.0 - pgamma).sqrt() * zlm.re,
//                             im: (1.0 + pgamma).sqrt() * zlm.im,
//                         },
//                     }
//                 })
//                 .build()
//                 .unwrap(),
//         )
//     }
// }
//
// impl IntoAmplitude for Zlm {
//     fn into_amplitude(self) -> Amplitude {
//         self.into_variable().cscalar().unwrap().into_amplitude()
//     }
// }

// #[derive(Clone)]
// pub struct Mass {
//     name: String,
//     particle_info: ParticleInfo,
// }
// impl VariableBuilder for Mass {
//     fn into_variable(self) -> Variable {
//         Variable::new(
//             &self.name,
//             move |vars: &VarMap| {
//                 let fs_p4s_lab = vars["Final State P4"].momenta_ref().unwrap();
//                 let resonance_p4_lab: &FourMomentum = &self
//                     .particle_info
//                     .resonance_indices
//                     .iter()
//                     .map(|i| &fs_p4s_lab[*i])
//                     .sum();
//                 FieldType::Scalar(resonance_p4_lab.m())
//             },
//             None,
//         )
//     }
// }
//
// #[derive(Clone)]
// pub struct KMatrixConstants {
//     pub g: Array2<f64>,
//     pub m: Array1<f64>,
//     pub c: Array2<f64>,
//     pub m1: Array1<f64>,
//     pub m2: Array1<f64>,
// }
//
// #[derive(Clone)]
// pub struct BarrierFactor {
//     name: String,
//     m: Array1<f64>,
//     m1: Array1<f64>,
//     m2: Array1<f64>,
//     l: usize,
//     n_resonances: usize,
//     n_channels: usize,
//     mass: Variable,
// }
//
// impl BarrierFactor {
//     fn new(
//         name: String,
//         constants: KMatrixConstants,
//         l: usize,
//         particle_info: &ParticleInfo,
//     ) -> BarrierFactor {
//         let n_resonances = constants.g.ncols();
//         let n_channels = constants.g.nrows();
//         let mass_name = format!("{}_mass", name.clone());
//         let mass = Mass {
//             name: mass_name.clone(),
//             particle_info: particle_info.clone(),
//         }
//         .into_variable();
//         BarrierFactor {
//             name,
//             m: constants.m,
//             m1: constants.m1,
//             m2: constants.m2,
//             l,
//             n_resonances,
//             n_channels,
//             mass,
//         }
//     }
// }
//
// impl BarrierFactor {
//     fn chi_plus(&self, s: f64, channel: usize) -> Complex64 {
//         (1.0 - ((&self.m1 + &self.m2) * (&self.m1 + &self.m2))[channel] / s).into()
//     }
//
//     fn chi_minus(&self, s: f64, channel: usize) -> Complex64 {
//         (1.0 - ((&self.m1 - &self.m2) * (&self.m1 - &self.m2))[channel] / s).into()
//     }
//
//     fn rho(&self, s: f64, channel: usize) -> Complex64 {
//         (self.chi_plus(s, channel) * self.chi_minus(s, channel)).sqrt()
//     }
//     fn z(&self, s: f64, channel: usize) -> Complex64 {
//         let q = self.rho(s, channel) * Complex64::sqrt(s.into()) / 2.0;
//         q * q / (0.1973 * 0.1973)
//     }
//     fn blatt_weisskopf(&self, s: f64, channel: usize) -> Complex64 {
//         let z = self.z(s, channel);
//         match self.l {
//             0 => 1.0.into(),
//             1 => ((2.0 * z) / (z + 1.0)).sqrt(),
//             2 => ((13.0 * z.powi(2)) / ((z - 3.0).powi(2) + 9.0 * z)).sqrt(),
//             3 => ((277.0 * z.powi(3)) / (z * (z - 15.0).powi(2) + 9.0 * (2.0 * z - 5.0).powi(2)))
//                 .sqrt(),
//             4 => ((12746.0 * z.powi(4)) / (z.powi(2) - 45.0 * z + 105.0).powi(2)
//                 + 25.0 * z * (2.0 * z - 21.0).powi(2))
//             .sqrt(),
//             l => panic!("L = {l} is not yet implemented"),
//         }
//     }
//     fn barrier_factor(&self, s: f64, channel: usize, resonance: usize) -> Complex64 {
//         let numerator = self.blatt_weisskopf(s, channel);
//         let denominator = self.blatt_weisskopf(self.m[resonance].powi(2), channel);
//         numerator / denominator
//     }
// }
// impl VariableBuilder for BarrierFactor {
//     fn into_variable(self) -> Variable {
//         let mass = self.mass.clone();
//         let mass_name = mass.name.clone();
//         Variable::new(
//             &self.name.clone(),
//             move |entry: &VarMap| {
//                 let s = entry[&*mass_name].scalar_ref().unwrap().powi(2);
//                 FieldType::CMatrix(Array2::from_shape_fn(
//                     (self.n_channels, self.n_resonances),
//                     |(i, a)| self.barrier_factor(s, i, a),
//                 ))
//             },
//             Some(vec![mass]),
//         )
//     }
// }
//
// #[derive(Clone)]
// pub struct AdlerZero {
//     pub s_0: f64,
//     pub s_norm: f64,
// }
//
// #[derive(Clone)]
// pub struct FrozenKMatrix {
//     name: String,
//     selected_channel: usize,
//     g: Array2<f64>,
//     m: Array1<f64>,
//     c: Array2<f64>,
//     m1: Array1<f64>,
//     m2: Array1<f64>,
//     l: usize,
//     n_resonances: usize,
//     n_channels: usize,
//     particle_info: ParticleInfo,
//     adler_zero: Option<AdlerZero>,
// }
//
// impl FrozenKMatrix {
//     pub fn new(
//         name: &str,
//         selected_channel: usize,
//         constants: KMatrixConstants,
//         l: usize,
//         particle_info: ParticleInfo,
//         adler_zero: Option<AdlerZero>,
//     ) -> FrozenKMatrix {
//         let n_resonances = constants.g.ncols();
//         let n_channels = constants.g.nrows();
//         FrozenKMatrix {
//             name: name.into(),
//             selected_channel,
//             g: constants.g,
//             m: constants.m,
//             c: constants.c,
//             m1: constants.m1,
//             m2: constants.m2,
//             l,
//             n_resonances,
//             n_channels,
//             particle_info,
//             adler_zero,
//         }
//     }
//
//     fn chi_plus(&self, s: f64, channel: usize) -> Complex64 {
//         (1.0 - ((&self.m1 + &self.m2) * (&self.m1 + &self.m2))[channel] / s).into()
//     }
//
//     fn chi_minus(&self, s: f64, channel: usize) -> Complex64 {
//         (1.0 - ((&self.m1 - &self.m2) * (&self.m1 - &self.m2))[channel] / s).into()
//     }
//
//     fn rho(&self, s: f64, channel: usize) -> Complex64 {
//         (self.chi_plus(s, channel) * self.chi_minus(s, channel)).sqrt()
//     }
//
//     fn c_matrix(&self, s: f64) -> Array2<Complex64> {
//         Array2::from_diag(&Array1::from_shape_fn(self.n_channels, |channel| {
//             self.rho(s, channel) / PI
//                 * ((self.chi_plus(s, channel) + self.rho(s, channel))
//                     / (self.chi_plus(s, channel) - self.rho(s, channel)))
//                 .ln()
//                 + self.chi_plus(s, channel) / PI
//                     * ((&self.m2 - &self.m1) / (&self.m1 + &self.m2))[channel]
//                     * Complex64::from((&self.m2 / &self.m1)[channel]).ln()
//         }))
//     }
// }
//
// impl VariableBuilder for FrozenKMatrix {
//     fn into_variable(self) -> Variable {
//         let mass_name = format!("{}_mass", self.name.clone());
//         let mass = Mass {
//             name: mass_name.clone(),
//             particle_info: self.particle_info.clone(),
//         }
//         .into_variable();
//         let bf_name = format!("{}_bf", self.name.clone());
//         let barrier_factor = BarrierFactor {
//             name: bf_name.clone(),
//             m: self.m.clone(),
//             m1: self.m1.clone(),
//             m2: self.m2.clone(),
//             l: self.l,
//             n_resonances: self.n_resonances,
//             n_channels: self.n_channels,
//             mass: mass.clone(),
//         }
//         .into_variable();
//
//         Variable::new(
//             &self.name.clone(),
//             move |entry: &VarMap| {
//                 let s = entry[&mass_name].scalar_ref().unwrap().powi(2);
//                 let bf = entry[&bf_name].cmatrix_ref().unwrap();
//                 let k_ija = Array3::from_shape_fn(
//                     (self.n_channels, self.n_channels, self.n_resonances),
//                     |(i, j, a)| {
//                         bf[[i, a]]
//                             * ((self.g[[i, a]] * self.g[[j, a]]) / (self.m[a].powi(2) - s)
//                                 + self.c[[i, j]])
//                             * bf[[j, a]]
//                     },
//                 );
//                 let k_mat = match self.adler_zero {
//                     Some(AdlerZero { s_0, s_norm }) => {
//                         Complex64::from((s - s_0) / s_norm) * k_ija.sum_axis(Axis(2))
//                     }
//                     None => k_ija.sum_axis(Axis(2)),
//                 };
//                 let c_mat = self.c_matrix(s);
//                 let i_mat: Array2<Complex64> = Array2::eye(self.n_channels);
//                 let ikc_mat = i_mat + k_mat * c_mat;
//                 let ikc_mat_inv = ikc_mat.inv().unwrap();
//                 FieldType::CVector(ikc_mat_inv.row(self.selected_channel).to_owned())
//             },
//             Some(vec![mass, barrier_factor]),
//         )
//     }
// }
//
// impl<'a> AmplitudeBuilder<'a> for FrozenKMatrix {
//     fn into_amplitude(self) -> Amplitude<'a> {
//         let ikc_inv_var = self.clone().into_variable();
//         let mass_name = ikc_inv_var.dependencies.clone().unwrap()[0].name.clone();
//         let bf_name = ikc_inv_var.dependencies.clone().unwrap()[1].name.clone();
//         let var_name = ikc_inv_var.name.clone();
//         let internal_parameters: Vec<String> = (0..self.n_resonances)
//             .map(|i| format!("beta_{i}"))
//             .collect();
//         let internal_parameters: Vec<&str> = internal_parameters.iter().map(|s| &**s).collect();
//         Amplitude::new(
//             &var_name.clone(),
//             move |pars: &ParMap, vars: &VarMap| {
//                 let s = vars[&*mass_name].scalar_ref().unwrap().powi(2);
//                 let bf = vars[&*bf_name].cmatrix_ref().unwrap();
//                 let ikc_inv_vec = vars[&*var_name].cvector_ref().unwrap();
//                 let betas = Array1::from_shape_fn(self.n_resonances, |i| {
//                     pars[&format!("beta_{i}")].value.cscalar().unwrap()
//                 });
//
//                 let p_ja = Array2::from_shape_fn((self.n_channels, self.n_resonances), |(j, a)| {
//                     ((betas[a] * self.g[[j, a]]) / (self.m[a].powi(2) - s)) * bf[[j, a]]
//                 });
//                 let p_vec = p_ja.sum_axis(Axis(1));
//
//                 Ok(ikc_inv_vec.dot(&p_vec))
//             },
//             Some(internal_parameters),
//             Some(vec![ikc_inv_var.clone()]),
//         )
//     }
// }
