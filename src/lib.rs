#![allow(dead_code)]
pub mod amplitude;
pub mod dataset;
pub mod four_momentum;
pub mod gluex;
pub mod variable;
// pub mod likelihood;
pub mod prelude {
    pub use crate::amplitude::{Amplitude, IntoAmplitude, ParMap, Parameter, ParameterValue};
    pub use crate::dataset::{
        CMatrix64, CScalar64, CVector64, DataType, Dataset, Entry, Matrix64, Momenta64, Momentum64,
        Scalar64, Vector64,
    };
    pub use crate::four_momentum::FourMomentum;
    // pub use crate::likelihood::ParallelExtendedMaximumLikelihood;
    pub use crate::variable::{
        CMatrixVariable, CMatrixVariableBuilder, CScalarVariable, CScalarVariableBuilder,
        CVectorVariable, CVectorVariableBuilder, IntoVariable, MatrixVariable,
        MatrixVariableBuilder, MomentaVariable, MomentaVariableBuilder, MomentumVariable,
        MomentumVariableBuilder, ScalarVariable, ScalarVariableBuilder, Variable, VectorVariable,
        VectorVariableBuilder,
    };
    pub use crate::{cpar, par, pars};
}
