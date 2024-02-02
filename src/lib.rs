#![allow(dead_code)]
pub mod dataset;
pub mod four_momentum;
pub mod gluex;
pub mod node;
// pub mod likelihood;
pub mod prelude {
    pub use crate::dataset::{
        extract_scalar, extract_vector, open_parquet, scalars_to_momentum, scalars_to_momentum_par,
        vectors_to_momenta, vectors_to_momenta_par, CMatrix64, CScalar64, CVector64, Dataset,
        DatasetError, Matrix64, ReadType, Scalar64, Vector64,
    };
    pub use crate::four_momentum::FourMomentum;
    pub use crate::node::{CNode, Dependent, Parameterized, Resolvable, SNode};
}
