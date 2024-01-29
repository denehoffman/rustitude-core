use std::collections::HashMap;

use crate::prelude::{CMatrix64, CScalar64, CVector64, Dataset, Matrix64, Scalar64, Vector64};

pub trait ResourceNode {
    fn eval(&self, ds: &mut Dataset) -> ();
    fn resources(&self) -> Vec<&dyn ResourceNode> {
        Vec::new()
    }
    fn resolve(&self, ds: &mut Dataset) {
        for resource in self.resources() {
            resource.resolve(ds);
        }
        self.eval(ds);
    }
}

pub trait ParameterizedNode {
    fn resources(&self) -> Vec<&dyn ResourceNode> {
        Vec::new()
    }
    fn get_external_par_name(&self, internal_par_name: String) -> &String;
    fn par(&self, internal_par_name: String, pars: HashMap<String, f64>) -> f64 {
        *pars
            .get(self.get_external_par_name(internal_par_name))
            .unwrap()
    }
    fn dependencies(&self) -> Vec<&dyn ParameterizedNode> {
        Vec::new()
    }
    fn resolve(&self, ds: &mut Dataset) {
        for resource in self.resources() {
            resource.resolve(ds);
        }
        for dependency in self.dependencies() {
            dependency.resolve(ds);
        }
    }
}

pub struct AddScalarNode<'a, 'b, T, U>
where
    T: 'a + ScalarNode,
    U: 'b + ScalarNode,
{
    a: &'a T,
    b: &'b U,
}

pub trait ScalarNode: ParameterizedNode {
    fn eval(&self, ds: &Dataset, pars: HashMap<String, f64>) -> Vec<Scalar64>;
}
pub trait CScalarNode: ParameterizedNode {
    fn eval(&self, ds: &Dataset, pars: HashMap<String, f64>) -> Vec<CScalar64>;
}
pub trait VectorNode: ParameterizedNode {
    fn eval(&self, ds: &Dataset, pars: HashMap<String, f64>) -> Vec<Vector64>;
}
pub trait CVectorNode: ParameterizedNode {
    fn eval(&self, ds: &Dataset, pars: HashMap<String, f64>) -> Vec<CVector64>;
}
pub trait MatrixNode: ParameterizedNode {
    fn eval(&self, ds: &Dataset, pars: HashMap<String, f64>) -> Vec<Matrix64>;
}
pub trait CMatrixNode: ParameterizedNode {
    fn eval(&self, ds: &Dataset, pars: HashMap<String, f64>) -> Vec<CMatrix64>;
}
