use std::hash::{Hash, Hasher};
use variantly::Variantly;

use crate::prelude::{CScalar64, Scalar64};

#[derive(Clone, Copy, Debug, Variantly)]
pub enum Parameter {
    #[variantly(rename = "scalar")]
    Scalar(ScalarParameter),
    #[variantly(rename = "cscalar")]
    CScalar(ComplexScalarParameter),
}

impl From<ScalarParameter> for Parameter {
    fn from(par: ScalarParameter) -> Self {
        Parameter::Scalar(par)
    }
}
impl From<ComplexScalarParameter> for Parameter {
    fn from(par: ComplexScalarParameter) -> Self {
        Parameter::CScalar(par)
    }
}

impl Parameter {
    fn name(&self) -> String {
        match self {
            Self::Scalar(par) => par.name.to_string(),
            Self::CScalar(par) => par.name.to_string(),
        }
    }
    fn value(&self) -> CScalar64 {
        match self {
            Self::Scalar(par) => par.value.into(),
            Self::CScalar(par) => par.value,
        }
    }
}

impl Hash for Parameter {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state)
    }
}

impl PartialEq for Parameter {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for Parameter {}

#[derive(Clone, Copy, Debug)]
pub struct ScalarParameter {
    name: &'static str,
    value: Scalar64,
}

#[derive(Clone, Copy, Debug)]
pub struct ComplexScalarParameter {
    name: &'static str,
    value: CScalar64,
}

// #[derive(Variantly, Clone, Copy, Debug)]
// pub enum ParameterValue {
//     Scalar(f64),
//     #[variantly(rename = "cscalar")]
//     CScalar(Complex64),
// }
//
// #[derive(Clone, Copy, Debug)]
// pub struct Parameter {
//     pub name: &'static str,
//     pub value: ParameterValue,
// }
//
// impl Hash for Parameter {
//     /// This ensures the hash lookup only depends on the name of the parameter, not its value
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         self.name.hash(state);
//     }
// }
//
// impl PartialEq for Parameter {
//     fn eq(&self, other: &Self) -> bool {
//         self.name == other.name
//     }
// }
//
// impl Eq for Parameter {}
//
// impl Parameter {
//     pub fn new(name: &'static str, value: ParameterValue) -> Parameter {
//         Parameter { name, value }
//     }
// }
//
// impl IntoAmplitude for Parameter {
//     fn into_amplitude(self) -> Amplitude {
//         AmplitudeBuilder::default()
//             .name(self.name)
//             .function(|pars: &ParMap, _vars: &Entry| {
//                 Ok(match pars.get("parameter").unwrap().value {
//                     ParameterValue::Scalar(val) => Complex64::from(val),
//                     ParameterValue::CScalar(val) => val,
//                 })
//             })
//             .internal_parameters(["parameter"])
//             .build()
//             .unwrap()
//     }
// }
