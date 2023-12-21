use std::{f64::consts::PI, fs::File};

use nalgebra::Vector3;
use ndarray::{Array1, Array2, Array3, Axis};
use ndarray_linalg::Inverse;
use num_complex::Complex64;
use polars::prelude::*;
use sphrs::{ComplexSH, Coordinates, SHEval};

use crate::prelude::*;

fn open_parquet(path: &str) -> DataFrame {
    let file = File::open(path).expect("Failed to open");
    ParquetReader::new(file).finish().expect("Failed to finish")
}

fn extract_scalar_field(column_name: &str, df: &DataFrame) -> Vec<FieldType> {
    return df
        .column(column_name)
        .unwrap_or_else(|_| panic!("No branch {column_name}"))
        .f32()
        .unwrap()
        .to_vec()
        .into_iter()
        .map(|x| FieldType::Scalar(x.unwrap().into()))
        .collect::<Vec<_>>();
}

fn extract_scalar(column_name: &str, df: &DataFrame) -> Vec<f64> {
    return df
        .column(column_name)
        .unwrap_or_else(|_| panic!("No branch {column_name}"))
        .f32()
        .unwrap()
        .to_vec()
        .into_iter()
        .map(|x| x.unwrap().into())
        .collect();
}

fn extract_array1(column_name: &str, df: &DataFrame) -> Vec<Array1<f64>> {
    return df
        .column(column_name)
        .unwrap_or_else(|_| panic!("No branch {column_name}"))
        .list()
        .unwrap()
        .into_iter()
        .map(|x| {
            Array1::from_vec(
                x.unwrap()
                    .f32()
                    .unwrap()
                    .to_vec()
                    .into_iter()
                    .map(|x| x.unwrap().into())
                    .collect(),
            )
        })
        .collect::<Vec<Array1<f64>>>();
}

pub fn open_gluex(path: &str, polarized: bool) -> Dataset {
    let dataframe = open_parquet(path);
    let col_n_fs = dataframe.column("NumFinalState").unwrap();
    let mut dataset = Dataset::new(col_n_fs.len());
    let n_fs = col_n_fs.i32().unwrap().into_iter().next().unwrap().unwrap() as usize;
    let e_beam = extract_scalar("E_Beam", &dataframe);
    let px_beam = extract_scalar("Px_Beam", &dataframe);
    let py_beam = extract_scalar("Py_Beam", &dataframe);
    let pz_beam = extract_scalar("Pz_Beam", &dataframe);
    let mut beam_p4: Vec<FieldType> = Vec::new();
    let mut eps: Vec<FieldType> = Vec::new();
    if polarized {
        for (((e, px), py), _) in e_beam
            .into_iter()
            .zip(px_beam.into_iter())
            .zip(py_beam.into_iter())
            .zip(pz_beam.into_iter())
        {
            beam_p4.push(FieldType::Momentum(FourMomentum::new(e, 0.0, 0.0, e)));
            eps.push(FieldType::Vector(Array1::from_vec(vec![px, py, 0.0])));
        }
        dataset.add_field("Beam P4", beam_p4, false);
        dataset.add_field("EPS", eps, false);
    } else {
        for (((e, px), py), pz) in e_beam
            .into_iter()
            .zip(px_beam.into_iter())
            .zip(py_beam.into_iter())
            .zip(pz_beam.into_iter())
        {
            beam_p4.push(FieldType::Momentum(FourMomentum::new(e, px, py, pz)))
        }
        dataset.add_field("Beam P4", beam_p4, false);
    }
    let weight = extract_scalar_field("Weight", &dataframe);
    let e_finalstate = extract_array1("E_FinalState", &dataframe);
    let px_finalstate = extract_array1("Px_FinalState", &dataframe);
    let py_finalstate = extract_array1("Py_FinalState", &dataframe);
    let pz_finalstate = extract_array1("Pz_FinalState", &dataframe);
    dataset.add_field("Weight", weight, false);
    let fs_p4: Vec<FieldType> = e_finalstate
        .iter()
        .zip(px_finalstate.iter())
        .zip(py_finalstate.iter())
        .zip(pz_finalstate.iter())
        .map(|(((e, px), py), pz)| {
            let mut momentum_vec = Vec::new();
            for i in 0..n_fs {
                momentum_vec.push(FourMomentum::new(e[i], px[i], py[i], pz[i]));
            }
            FieldType::MomentumVec(momentum_vec)
        })
        .collect();
    dataset.add_field("Final State P4", fs_p4, false);
    dataset
}

#[derive(Clone)]
pub struct ParticleInfo {
    pub recoil_index: usize,
    pub daughter_index: usize,
    pub resonance_indices: Vec<usize>,
}

#[derive(Clone)]
pub struct Ylm {
    pub l: usize,
    pub m: isize,
    pub particle_info: ParticleInfo,
}

impl VariableBuilder for Ylm {
    fn to_variable(self) -> Variable {
        Variable::new(
            format!("Y {} {}", self.l, self.m),
            move |entry: &VarMap| {
                let beam_p4_lab = entry["Beam P4"].momentum().unwrap();
                let fs_p4s_lab = entry["Final State P4"].momenta().unwrap();
                let fs_p4_lab = &fs_p4s_lab.iter().sum();
                let recoil_p4_lab = &fs_p4s_lab[self.particle_info.recoil_index];
                let resonance_p4_lab: &FourMomentum = &self
                    .particle_info
                    .resonance_indices
                    .iter()
                    .map(|i| &fs_p4s_lab[*i])
                    .sum();
                let daughter_p4_lab = &fs_p4s_lab[self.particle_info.daughter_index];

                let beam_p4 = &beam_p4_lab.boost_along(fs_p4_lab);
                let recoil_p4 = &recoil_p4_lab.boost_along(fs_p4_lab);
                let resonance_p4 = &resonance_p4_lab.boost_along(fs_p4_lab);
                let daughter_p4 = &daughter_p4_lab.boost_along(fs_p4_lab);

                let recoil_p4_res = &recoil_p4.boost_along(resonance_p4);
                let daughter_p4_res = &daughter_p4.boost_along(resonance_p4);

                let z = -1.0 * recoil_p4_res.momentum().normalize();
                let y = beam_p4
                    .momentum()
                    .cross(&(-1.0 * recoil_p4.momentum()))
                    .normalize();
                let x = y.cross(&z);

                let daughter_p3_res = daughter_p4_res.momentum();

                let p = Coordinates::cartesian(
                    daughter_p3_res.dot(&x),
                    daughter_p3_res.dot(&y),
                    daughter_p3_res.dot(&z),
                );
                FieldType::CScalar(ComplexSH::Spherical.eval(self.l as i64, self.m as i64, &p))
            },
            None,
        )
    }
}

impl AmplitudeBuilder for Ylm {
    fn internal_parameter_names(&self) -> Option<Vec<String>> {
        None
    }
    fn to_amplitude(self) -> Amplitude {
        let ylm_var = self.to_variable();
        let var_name = ylm_var.name.to_string();
        Amplitude::new(var_name.clone(), move |_pars: &ParMap, vars: &VarMap| {
            Ok(*(vars[&*var_name].cscalar()?))
        })
        .with_deps(vec![ylm_var.clone()])
    }
}

pub enum Reflectivity {
    Positive,
    Negative,
}
pub struct Zlm {
    pub l: usize,
    pub m: isize,
    pub r: Reflectivity,
    pub particle_info: ParticleInfo,
}
impl VariableBuilder for Zlm {
    fn to_variable(self) -> Variable {
        Variable::new(
            format!(
                "Z {} {} {}",
                self.l,
                self.m,
                match self.r {
                    Reflectivity::Positive => "+",
                    Reflectivity::Negative => "-",
                }
            ),
            move |entry: &VarMap| {
                let beam_p4_lab = entry["Beam P4"].momentum().unwrap();
                let fs_p4s_lab = entry["Final State P4"].momenta().unwrap();
                let fs_p4_lab = &fs_p4s_lab.iter().sum();
                let recoil_p4_lab = &fs_p4s_lab[self.particle_info.recoil_index];
                let resonance_p4_lab: &FourMomentum = &self
                    .particle_info
                    .resonance_indices
                    .iter()
                    .map(|i| &fs_p4s_lab[*i])
                    .sum();
                let daughter_p4_lab = &fs_p4s_lab[self.particle_info.daughter_index];

                let beam_p4 = &beam_p4_lab.boost_along(fs_p4_lab);
                let recoil_p4 = &recoil_p4_lab.boost_along(fs_p4_lab);
                let resonance_p4 = &resonance_p4_lab.boost_along(fs_p4_lab);
                let daughter_p4 = &daughter_p4_lab.boost_along(fs_p4_lab);

                let recoil_p4_res = &recoil_p4.boost_along(resonance_p4);
                let daughter_p4_res = &daughter_p4.boost_along(resonance_p4);

                let z = -1.0 * recoil_p4_res.momentum().normalize();
                let y = beam_p4
                    .momentum()
                    .cross(&(-1.0 * recoil_p4.momentum()))
                    .normalize();
                let x = y.cross(&z);

                let daughter_p3_res = daughter_p4_res.momentum();

                let p = Coordinates::cartesian(
                    daughter_p3_res.dot(&x),
                    daughter_p3_res.dot(&y),
                    daughter_p3_res.dot(&z),
                );
                let ylm = ComplexSH::Spherical.eval(self.l as i64, self.m as i64, &p);

                // Polarization
                let eps = Vector3::from_vec(entry["EPS"].vector().unwrap().to_vec());
                let big_phi = y
                    .dot(&eps)
                    .atan2(beam_p4.momentum().normalize().dot(&eps.cross(&y)));
                let phase = Complex64::cis(-big_phi);
                let pgamma = eps.norm();

                let zlm = ylm * phase;

                let res = match self.r {
                    Reflectivity::Positive => Complex64 {
                        re: (1.0 + pgamma).sqrt() * zlm.re,
                        im: (1.0 - pgamma).sqrt() * zlm.im,
                    },
                    Reflectivity::Negative => Complex64 {
                        re: (1.0 - pgamma).sqrt() * zlm.re,
                        im: (1.0 + pgamma).sqrt() * zlm.im,
                    },
                };
                FieldType::CScalar(res)
            },
            None,
        )
    }
}
impl AmplitudeBuilder for Zlm {
    fn internal_parameter_names(&self) -> Option<Vec<String>> {
        None
    }
    fn to_amplitude(self) -> Amplitude {
        let zlm_var = self.to_variable();
        let var_name = zlm_var.name.to_string();
        Amplitude::new(var_name.clone(), move |_pars: &ParMap, vars: &VarMap| {
            Ok(*(vars[&*var_name].cscalar()?))
        })
        .with_deps(vec![zlm_var.clone()])
    }
}

#[derive(Clone)]
pub struct AdlerZero {
    pub s_0: f64,
    pub s_norm: f64,
}

#[derive(Clone)]
pub struct FrozenKMatrix {
    name: String,
    selected_channel: usize,
    g: Array2<f64>,
    m: Array1<f64>,
    c: Array2<f64>,
    m1: Array1<f64>,
    m2: Array1<f64>,
    l: usize,
    n_resonances: usize,
    n_channels: usize,
    particle_info: ParticleInfo,
    adler_zero: Option<AdlerZero>,
}

impl FrozenKMatrix {
    pub fn new(
        name: &str,
        selected_channel: usize,
        g: Array2<f64>,
        m: Array1<f64>,
        c: Array2<f64>,
        m1: Array1<f64>,
        m2: Array1<f64>,
        l: usize,
        particle_info: ParticleInfo,
        adler_zero: Option<AdlerZero>,
    ) -> FrozenKMatrix {
        let n_resonances = g.ncols();
        let n_channels = g.nrows();
        FrozenKMatrix {
            name: name.into(),
            selected_channel,
            g,
            m,
            c,
            m1,
            m2,
            l,
            n_resonances,
            n_channels,
            particle_info,
            adler_zero,
        }
    }

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
        match self.l {
            0 => 1.0.into(),
            1 => ((2.0 * z) / (z + 1.0)).sqrt(),
            2 => ((13.0 * z.powi(2)) / ((z - 3.0).powi(2) + 9.0 * z)).sqrt(),
            3 => ((277.0 * z.powi(3)) / (z * (z - 15.0).powi(2) + 9.0 * (2.0 * z - 5.0).powi(2)))
                .sqrt(),
            4 => ((12746.0 * z.powi(4)) / (z.powi(2) - 45.0 * z + 105.0).powi(2)
                + 25.0 * z * (2.0 * z - 21.0).powi(2))
            .sqrt(),
            l => panic!("L = {} is not yet implemented", l),
        }
    }
    fn barrier_factor(&self, s: f64, channel: usize, resonance: usize) -> Complex64 {
        let numerator = self.blatt_weisskopf(s, channel);
        let denominator = self.blatt_weisskopf(self.m[resonance].powi(2), channel);
        numerator / denominator
    }
    fn k_matrix(&self, s: f64) -> Array2<Complex64> {
        let k_ija = Array3::from_shape_fn(
            (self.n_channels, self.n_channels, self.n_resonances),
            |(i, j, a)| {
                self.barrier_factor(s, i, a)
                    * ((self.g[[i, a]] * self.g[[j, a]]) / (self.m[a].powi(2) - s) + self.c[[i, j]])
                    * self.barrier_factor(s, j, a)
            },
        );
        match self.adler_zero {
            Some(AdlerZero { s_0, s_norm }) => {
                Complex64::from((s - s_0) / s_norm) * k_ija.sum_axis(Axis(2))
            }
            None => k_ija.sum_axis(Axis(2)),
        }
    }
}
impl VariableBuilder for FrozenKMatrix {
    fn to_variable(self) -> Variable {
        Variable::new(
            self.name.clone(),
            move |entry: &VarMap| {
                let fs_p4s_lab = entry["Final State P4"].momenta().unwrap();
                let resonance_p4_lab: &FourMomentum = &self
                    .particle_info
                    .resonance_indices
                    .iter()
                    .map(|i| &fs_p4s_lab[*i])
                    .sum();
                let s = resonance_p4_lab.m2();
                let k_mat = self.k_matrix(s);
                let c_mat = self.c_matrix(s);
                let i_mat: Array2<Complex64> = Array2::eye(self.n_channels);
                let ikc_mat = i_mat + k_mat * c_mat;
                let ikc_mat_inv = ikc_mat.inv().unwrap();
                FieldType::CVector(ikc_mat_inv.row(self.selected_channel).to_owned())
            },
            None,
        )
    }
}

impl AmplitudeBuilder for FrozenKMatrix {
    fn internal_parameter_names(&self) -> Option<Vec<String>> {
        Some(
            (0..self.n_resonances)
                .into_iter()
                .map(|i| format!("beta_{}", i))
                .collect(),
        )
    }
    fn to_amplitude(self) -> Amplitude {
        let ikc_inv_var = self.clone().to_variable();
        let var_name = ikc_inv_var.name.to_string();
        Amplitude::new(var_name.clone(), move |pars: &ParMap, vars: &VarMap| {
            let fs_p4s_lab = vars["Final State P4"].momenta().unwrap();
            let resonance_p4_lab: &FourMomentum = &self
                .particle_info
                .resonance_indices
                .iter()
                .map(|i| &fs_p4s_lab[*i])
                .sum();
            let s = resonance_p4_lab.m2();
            let ikc_inv_vec = vars[&*var_name].cvector()?;
            let betas = Array1::from_shape_fn(self.n_resonances, |i| {
                pars[&format!("beta_{}", i)].cscalar().unwrap()
            });

            let p_ja = Array2::from_shape_fn((self.n_channels, self.n_resonances), |(j, a)| {
                ((betas[a] * self.g[[j, a]]) / (self.m[a].powi(2) - s))
                    * self.barrier_factor(s, j, a)
            });
            let p_vec = p_ja.sum_axis(Axis(1));

            Ok(ikc_inv_vec.dot(&p_vec))
        })
        .with_deps(vec![ikc_inv_var.clone()])
    }
}
