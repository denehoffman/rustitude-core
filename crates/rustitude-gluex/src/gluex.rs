use std::{f64::consts::PI, fmt::Display};

use rustitude::prelude::*;

use nalgebra::{ComplexField, SMatrix, SVector};
use num_complex::Complex64;
use rayon::prelude::*;
use sphrs::SHEval;
use sphrs::{ComplexSH, Coordinates};

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

pub struct Ylm(Wave, Vec<Complex64>);
impl Ylm {
    pub fn new(wave: Wave) -> Self {
        Self(wave, Vec::default())
    }
}
impl Node for Ylm {
    fn parameters(&self) -> Option<Vec<String>> {
        None
    }

    fn precalculate(&mut self, dataset: &Dataset) {
        self.1 = dataset
            .par_iter()
            .map(|event| {
                let resonance = event.daughter_p4s[0] + event.daughter_p4s[1];
                let p1 = event.daughter_p4s[0];
                let recoil_res = event.recoil_p4.boost_along(&resonance);
                let p1_res = p1.boost_along(&resonance);
                let z = -1.0 * recoil_res.momentum().normalize();
                let y = event
                    .beam_p4
                    .momentum()
                    .cross(&(-1.0 * event.recoil_p4.momentum()));
                let x = y.cross(&z);
                let p1_vec = p1_res.momentum();
                let p = Coordinates::cartesian(p1_vec.dot(&x), p1_vec.dot(&y), p1_vec.dot(&z));
                ComplexSH::Spherical.eval(self.0.l(), self.0.m(), &p)
            })
            .collect();
    }

    fn calculate(&self, _parameters: &[f64], event: &Event) -> Complex64 {
        self.1[event.index]
    }
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

pub struct ReZlm(pub Wave, pub Reflectivity, Vec<f64>);
impl ReZlm {
    pub fn new(wave: Wave, reflectivity: Reflectivity) -> Self {
        Self(wave, reflectivity, Vec::default())
    }
}
impl Node for ReZlm {
    fn precalculate(&mut self, dataset: &Dataset) {
        self.2 = dataset
            .par_iter()
            .map(|event| {
                let resonance = event.daughter_p4s[0] + event.daughter_p4s[1];
                let p1 = event.daughter_p4s[0];
                let recoil_res = event.recoil_p4.boost_along(&resonance);
                let p1_res = p1.boost_along(&resonance);
                let z = -1.0 * recoil_res.momentum().normalize();
                let y = event
                    .beam_p4
                    .momentum()
                    .cross(&(-1.0 * event.recoil_p4.momentum()));
                let x = y.cross(&z);
                let p1_vec = p1_res.momentum();
                let p = Coordinates::cartesian(p1_vec.dot(&x), p1_vec.dot(&y), p1_vec.dot(&z));
                let ylm = ComplexSH::Spherical.eval(self.0.l(), self.0.m(), &p);
                let big_phi = y.dot(&event.eps).atan2(
                    event
                        .beam_p4
                        .momentum()
                        .normalize()
                        .dot(&event.eps.cross(&y)),
                );
                let pgamma = event.eps.norm();

                let phase = Complex64::cis(-big_phi);
                let zlm = ylm * phase;
                match self.1 {
                    Reflectivity::Positive => (1.0 + pgamma).sqrt() * zlm.re,
                    Reflectivity::Negative => (1.0 - pgamma).sqrt() * zlm.re,
                }
            })
            .collect()
    }
    fn calculate(&self, _parameters: &[f64], event: &Event) -> Complex64 {
        self.2[event.index].into()
    }
    fn parameters(&self) -> Option<Vec<String>> {
        None
    }
}

pub struct ImZlm(pub Wave, pub Reflectivity, Vec<f64>);
impl ImZlm {
    pub fn new(wave: Wave, reflectivity: Reflectivity) -> Self {
        Self(wave, reflectivity, Vec::default())
    }
}
impl Node for ImZlm {
    fn precalculate(&mut self, dataset: &Dataset) {
        self.2 = dataset
            .par_iter()
            .map(|event| {
                let resonance = event.daughter_p4s[0] + event.daughter_p4s[1];
                let p1 = event.daughter_p4s[0];
                let recoil_res = event.recoil_p4.boost_along(&resonance);
                let p1_res = p1.boost_along(&resonance);
                let z = -1.0 * recoil_res.momentum().normalize();
                let y = event
                    .beam_p4
                    .momentum()
                    .cross(&(-1.0 * event.recoil_p4.momentum()));
                let x = y.cross(&z);
                let p1_vec = p1_res.momentum();
                let p = Coordinates::cartesian(p1_vec.dot(&x), p1_vec.dot(&y), p1_vec.dot(&z));
                let ylm = ComplexSH::Spherical.eval(self.0.l(), self.0.m(), &p);
                let big_phi = y.dot(&event.eps).atan2(
                    event
                        .beam_p4
                        .momentum()
                        .normalize()
                        .dot(&event.eps.cross(&y)),
                );
                let pgamma = event.eps.norm();

                let phase = Complex64::cis(-big_phi);
                let zlm = ylm * phase;
                match self.1 {
                    Reflectivity::Positive => (1.0 - pgamma).sqrt() * zlm.im,
                    Reflectivity::Negative => (1.0 + pgamma).sqrt() * zlm.im,
                }
            })
            .collect()
    }
    fn calculate(&self, _parameters: &[f64], event: &Event) -> Complex64 {
        self.2[event.index].into()
    }
    fn parameters(&self) -> Option<Vec<String>> {
        None
    }
}

#[derive(Clone, Copy)]
pub struct AdlerZero {
    pub s_0: f64,
    pub s_norm: f64,
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

impl<const C: usize, const R: usize> KMatrixConstants<C, R> {
    fn chi_plus(s: f64, m1: f64, m2: f64) -> f64 {
        1.0 - ((m1 + m2) * (m1 + m2)) / s
    }

    fn chi_minus(s: f64, m1: f64, m2: f64) -> f64 {
        1.0 - ((m1 - m2) * (m1 - m2)) / s
    }

    fn rho(s: f64, m1: f64, m2: f64) -> Complex64 {
        Complex64::from(Self::chi_plus(s, m1, m2) * Self::chi_minus(s, m1, m2)).sqrt()
    }
    fn c_matrix(&self, s: f64) -> SMatrix<Complex64, C, C> {
        SMatrix::from_diagonal(&SVector::from_fn(|i, _| {
            Self::rho(s, self.m1s[i], self.m2s[i]) / PI
                * ((Self::chi_plus(s, self.m1s[i], self.m2s[i])
                    + Self::rho(s, self.m1s[i], self.m2s[i]))
                    / (Self::chi_plus(s, self.m1s[i], self.m2s[i])
                        - Self::rho(s, self.m1s[i], self.m2s[i])))
                .ln()
                + Self::chi_plus(s, self.m1s[i], self.m2s[i]) / PI
                    * ((self.m2s[i] - self.m1s[i]) / (self.m1s[i] + self.m2s[i]))
                    * (self.m2s[i] / self.m1s[i]).ln()
        }))
    }
    fn z(s: f64, m1: f64, m2: f64) -> Complex64 {
        Self::rho(s, m1, m2).powi(2) * s / (2.0 * 0.1973 * 0.1973)
    }
    fn blatt_weisskopf(s: f64, m1: f64, m2: f64, l: usize) -> Complex64 {
        let z = Self::z(s, m1, m2);
        match l {
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
    fn barrier_factor(s: f64, m1: f64, m2: f64, mr: f64, l: usize) -> Complex64 {
        Self::blatt_weisskopf(s, m1, m2, l) / Self::blatt_weisskopf(mr.powi(2), m1, m2, l)
    }
    fn barrier_matrix(&self, s: f64) -> SMatrix<Complex64, C, R> {
        SMatrix::from_fn(|i, a| {
            Self::barrier_factor(s, self.m1s[i], self.m2s[i], self.mrs[a], self.l)
        })
    }

    fn k_matrix(&self, s: f64) -> SMatrix<Complex64, C, C> {
        let bf = self.barrier_matrix(s);
        SMatrix::from_fn(|i, j| {
            (0..R)
                .map(|a| {
                    bf[(i, a)]
                        * bf[(j, a)]
                        * (self.g[(i, a)] * self.g[(j, a)] / (self.mrs[a].powi(2) - s)
                            + self.c[(i, j)])
                })
                .sum::<Complex64>()
                * self.adler_zero.map_or(1.0, |az| (s - az.s_0) / az.s_norm)
        })
    }
    fn ikc_inv(&self, s: f64, channel: usize) -> SVector<Complex64, C> {
        let c_mat = self.c_matrix(s);
        let i_mat = SMatrix::<Complex64, C, C>::identity();
        let k_mat = self.k_matrix(s);
        let ikc_mat = i_mat + k_mat * c_mat;
        let ikc_inv_mat = ikc_mat.try_inverse().unwrap();
        ikc_inv_mat.row(channel).transpose()
    }

    fn p_vector(
        betas: &SVector<Complex64, R>,
        pvector_constants: &SMatrix<Complex64, C, R>,
    ) -> SVector<Complex64, C> {
        SVector::<Complex64, C>::from_fn(|j, _| {
            (0..R).map(|a| betas[a] * pvector_constants[(j, a)]).sum()
        })
    }

    pub fn calculate_k_matrix(
        betas: &SVector<Complex64, R>,
        ikc_inv_vec: &SVector<Complex64, C>,
        pvector_constants_mat: &SMatrix<Complex64, C, R>,
    ) -> Complex64 {
        ikc_inv_vec.dot(&Self::p_vector(betas, pvector_constants_mat))
    }
}
pub struct KMatrixF0(
    usize,
    KMatrixConstants<5, 5>,
    Vec<(SVector<Complex64, 5>, SMatrix<Complex64, 5, 5>)>,
);
#[rustfmt::skip]
impl KMatrixF0 {
    pub fn new(channel: usize) -> Self {
        Self(channel,
             KMatrixConstants {
                g: SMatrix::<f64, 5, 5>::new(
                     0.74987, -0.01257, 0.02736, -0.15102,  0.36103,
                     0.06401,  0.00204, 0.77413,  0.50999,  0.13112,
                    -0.23417, -0.01032, 0.72283,  0.11934,  0.36792,
                     0.01570,  0.26700, 0.09214,  0.02742, -0.04025,
                    -0.14242,  0.22780, 0.15981,  0.16272, -0.17397,
                ),
                c: SMatrix::<f64, 5, 5>::new(
                     0.03728, 0.00000, -0.01398, -0.02203,  0.01397,
                     0.00000, 0.00000,  0.00000,  0.00000,  0.00000,
                    -0.01398, 0.00000,  0.02349,  0.03101, -0.04003,
                    -0.02203, 0.00000,  0.03101, -0.13769, -0.06722,
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
            },
            Vec::default())
    }
}

impl Node for KMatrixF0 {
    fn precalculate(&mut self, dataset: &Dataset) {
        self.2 = dataset
            .par_iter()
            .map(|event| {
                let s = (event.daughter_p4s[0] + event.daughter_p4s[1]).m2();
                let barrier_mat = self.1.barrier_matrix(s);
                let pvector_constants = SMatrix::<Complex64, 5, 5>::from_fn(|i, a| {
                    barrier_mat[(i, a)] * self.1.g[(i, a)] / (self.1.mrs[a].powi(2) - s)
                });
                (self.1.ikc_inv(s, self.0), pvector_constants)
            })
            .collect();
    }
    fn calculate(&self, parameters: &[f64], event: &Event) -> Complex64 {
        let betas = SVector::<Complex64, 5>::new(
            Complex64::new(parameters[0], parameters[1]),
            Complex64::new(parameters[2], parameters[3]),
            Complex64::new(parameters[4], parameters[5]),
            Complex64::new(parameters[6], parameters[7]),
            Complex64::new(parameters[8], parameters[9]),
        );
        let (ikc_inv_vec, pvector_constants_mat) = self.2[event.index];
        KMatrixConstants::calculate_k_matrix(&betas, &ikc_inv_vec, &pvector_constants_mat)
    }
    fn parameters(&self) -> Option<Vec<String>> {
        Some(vec![
            "f0_500 re".to_string(),
            "f0_500 im".to_string(),
            "f0_980 re".to_string(),
            "f0_980 im".to_string(),
            "f0_1370 re".to_string(),
            "f0_1370 im".to_string(),
            "f0_1500 re".to_string(),
            "f0_1500 im".to_string(),
            "f0_1710 re".to_string(),
            "f0_1710 im".to_string(),
        ])
    }
}
pub struct KMatrixF2(
    usize,
    KMatrixConstants<4, 4>,
    Vec<(SVector<Complex64, 4>, SMatrix<Complex64, 4, 4>)>,
);
#[rustfmt::skip]
impl KMatrixF2 {
    pub fn new(channel: usize) -> Self {
        Self(channel,
             KMatrixConstants {
                g: SMatrix::<f64, 4, 4>::new(
                     0.40033, 0.01820, -0.06709, -0.49924,
                     0.15479, 0.17300,  0.22941,  0.19295,
                    -0.08900, 0.32393, -0.43133,  0.27975, 
                    -0.00113, 0.15256,  0.23721, -0.03987,
                ),
                c: SMatrix::<f64, 4, 4>::new(
                    -0.04319, 0.00000,  0.00984,  0.01028,
                     0.00000, 0.00000,  0.00000,  0.00000,
                     0.00984, 0.00000, -0.07344,  0.05533,
                     0.01028, 0.00000,  0.05533, -0.05183,
                ),
                m1s: [0.13498, 0.26995, 0.49368, 0.54786],
                m2s: [0.13498, 0.26995, 0.49761, 0.54786],
                mrs: [1.15299, 1.48359, 1.72923, 1.96700],
                adler_zero: None,
                l: 2,
            },
            Vec::default())
    }
}

impl Node for KMatrixF2 {
    fn precalculate(&mut self, dataset: &Dataset) {
        self.2 = dataset
            .par_iter()
            .map(|event| {
                let s = (event.daughter_p4s[0] + event.daughter_p4s[1]).m2();
                let barrier_mat = self.1.barrier_matrix(s);
                let pvector_constants = SMatrix::<Complex64, 4, 4>::from_fn(|i, a| {
                    barrier_mat[(i, a)] * self.1.g[(i, a)] / (self.1.mrs[a].powi(2) - s)
                });
                (self.1.ikc_inv(s, self.0), pvector_constants)
            })
            .collect();
    }
    fn calculate(&self, parameters: &[f64], event: &Event) -> Complex64 {
        let betas = SVector::<Complex64, 4>::new(
            Complex64::new(parameters[0], parameters[1]),
            Complex64::new(parameters[2], parameters[3]),
            Complex64::new(parameters[4], parameters[5]),
            Complex64::new(parameters[6], parameters[7]),
        );
        let (ikc_inv_vec, pvector_constants_mat) = self.2[event.index];
        KMatrixConstants::calculate_k_matrix(&betas, &ikc_inv_vec, &pvector_constants_mat)
    }
    fn parameters(&self) -> Option<Vec<String>> {
        Some(vec![
            "f2_1270 re".to_string(),
            "f2_1270 im".to_string(),
            "f2_1525 re".to_string(),
            "f2_1525 im".to_string(),
            "f2_1810 re".to_string(),
            "f2_1810 im".to_string(),
            "f2_1950 re".to_string(),
            "f2_1950 im".to_string(),
        ])
    }
}

pub struct KMatrixA0(
    usize,
    KMatrixConstants<2, 2>,
    Vec<(SVector<Complex64, 2>, SMatrix<Complex64, 2, 2>)>,
);
#[rustfmt::skip]
impl KMatrixA0 {
    pub fn new(channel: usize) -> Self {
        Self(channel,
             KMatrixConstants {
                g: SMatrix::<f64, 2, 2>::new(
                     0.43215, 0.19000,
                    -0.28825, 0.43372
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
            },
            Vec::default())
    }
}

impl Node for KMatrixA0 {
    fn precalculate(&mut self, dataset: &Dataset) {
        self.2 = dataset
            .par_iter()
            .map(|event| {
                let s = (event.daughter_p4s[0] + event.daughter_p4s[1]).m2();
                let barrier_mat = self.1.barrier_matrix(s);
                let pvector_constants = SMatrix::<Complex64, 2, 2>::from_fn(|i, a| {
                    barrier_mat[(i, a)] * self.1.g[(i, a)] / (self.1.mrs[a].powi(2) - s)
                });
                (self.1.ikc_inv(s, self.0), pvector_constants)
            })
            .collect();
    }
    fn calculate(&self, parameters: &[f64], event: &Event) -> Complex64 {
        let betas = SVector::<Complex64, 2>::new(
            Complex64::new(parameters[0], parameters[1]),
            Complex64::new(parameters[2], parameters[3]),
        );
        let (ikc_inv_vec, pvector_constants_mat) = self.2[event.index];
        KMatrixConstants::calculate_k_matrix(&betas, &ikc_inv_vec, &pvector_constants_mat)
    }
    fn parameters(&self) -> Option<Vec<String>> {
        Some(vec![
            "a0_980 re".to_string(),
            "a0_980 im".to_string(),
            "a0_1450 re".to_string(),
            "a0_1450 im".to_string(),
        ])
    }
}

pub struct KMatrixA2(
    usize,
    KMatrixConstants<3, 2>,
    Vec<(SVector<Complex64, 3>, SMatrix<Complex64, 3, 2>)>,
);
#[rustfmt::skip]
impl KMatrixA2 {
    pub fn new(channel: usize) -> Self {
        Self(channel,
             KMatrixConstants {
                g: SMatrix::<f64, 3, 2>::new(
                     0.30073, 0.68567,
                     0.21426, 0.12543, 
                    -0.09162, 0.00184
                ),
                c: SMatrix::<f64, 3, 3>::new(
                    -0.40184,  0.00033, -0.08707,
                     0.00033, -0.21416, -0.06193,
                    -0.08707, -0.06193, -0.17435,
                ),
                m1s: [0.13498, 0.49368, 0.13498],
                m2s: [0.54786, 0.49761, 0.95778],
                mrs: [1.30080, 1.75351],
                adler_zero: None,
                l: 2,
            },
            Vec::default())
    }
}

impl Node for KMatrixA2 {
    fn precalculate(&mut self, dataset: &Dataset) {
        self.2 = dataset
            .par_iter()
            .map(|event| {
                let s = (event.daughter_p4s[0] + event.daughter_p4s[1]).m2();
                let barrier_mat = self.1.barrier_matrix(s);
                let pvector_constants = SMatrix::<Complex64, 3, 2>::from_fn(|i, a| {
                    barrier_mat[(i, a)] * self.1.g[(i, a)] / (self.1.mrs[a].powi(2) - s)
                });
                (self.1.ikc_inv(s, self.0), pvector_constants)
            })
            .collect();
    }
    fn calculate(&self, parameters: &[f64], event: &Event) -> Complex64 {
        let betas = SVector::<Complex64, 2>::new(
            Complex64::new(parameters[0], parameters[1]),
            Complex64::new(parameters[2], parameters[3]),
        );
        let (ikc_inv_vec, pvector_constants_mat) = self.2[event.index];
        KMatrixConstants::calculate_k_matrix(&betas, &ikc_inv_vec, &pvector_constants_mat)
    }
    fn parameters(&self) -> Option<Vec<String>> {
        Some(vec![
            "a2_1320 re".to_string(),
            "a2_1320 im".to_string(),
            "a2_1700 re".to_string(),
            "a2_1700 im".to_string(),
        ])
    }
}

pub struct KMatrixPi1(
    usize,
    KMatrixConstants<2, 1>,
    Vec<(SVector<Complex64, 2>, SMatrix<Complex64, 2, 1>)>,
);
#[rustfmt::skip]
impl KMatrixPi1 {
    pub fn new(channel: usize) -> Self {
        Self(channel,
             KMatrixConstants {
                g: SMatrix::<f64, 2, 1>::new(
                    0.80564,
                    1.04595
                ),
                c: SMatrix::<f64, 2, 2>::new(
                    1.05000,  0.15163,
                    0.15163, -0.24611,
                ),
                m1s: [0.13498, 0.13498],
                m2s: [0.54786, 0.95778],
                mrs: [1.38552],
                adler_zero: None,
                l: 1,
            },
            Vec::default())
    }
}

impl Node for KMatrixPi1 {
    fn precalculate(&mut self, dataset: &Dataset) {
        self.2 = dataset
            .par_iter()
            .map(|event| {
                let s = (event.daughter_p4s[0] + event.daughter_p4s[1]).m2();
                let barrier_mat = self.1.barrier_matrix(s);
                let pvector_constants = SMatrix::<Complex64, 2, 1>::from_fn(|i, a| {
                    barrier_mat[(i, a)] * self.1.g[(i, a)] / (self.1.mrs[a].powi(2) - s)
                });
                (self.1.ikc_inv(s, self.0), pvector_constants)
            })
            .collect();
    }
    fn calculate(&self, parameters: &[f64], event: &Event) -> Complex64 {
        let betas = SVector::<Complex64, 1>::new(Complex64::new(parameters[0], parameters[1]));
        let (ikc_inv_vec, pvector_constants_mat) = self.2[event.index];
        KMatrixConstants::calculate_k_matrix(&betas, &ikc_inv_vec, &pvector_constants_mat)
    }
    fn parameters(&self) -> Option<Vec<String>> {
        Some(vec!["pi1_1600 re".to_string(), "pi1_1600 im".to_string()])
    }
}
