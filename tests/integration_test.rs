use anyinput::anyinput;
use ndarray::prelude::*;
use polars::prelude::*;
use rayon::prelude::*;
use rustitude::prelude::*;
use std::path::PathBuf;

#[anyinput]
pub fn open_gluex(
    path: AnyPath,
    polarized: bool,
    mass_bin: (f64, f64),
) -> Result<Dataset, DatasetError> {
    let total_dataframe = open_parquet(path.to_str().unwrap()).expect("Read error");
    let dataframe = total_dataframe
        .lazy()
        .filter(
            col("M_FinalState")
                .gt(mass_bin.0)
                .and(col("M_FinalState").lt(mass_bin.1)),
        )
        .collect()
        .unwrap();
    let weight = extract_scalar("Weight", &dataframe, ReadType::F32);
    let mut dataset = Dataset::from_size(dataframe.height(), Some(weight));
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

#[test]
fn load_dataset() {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("resources/data.parquet");
    let ds = open_gluex(d, true, (1.0, 1.025));
    assert!(ds.is_ok());
}
