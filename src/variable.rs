use rayon::prelude::*;
use std::sync::Arc;
use variantly::Variantly;

use anyinput::anyinput;
use derive_builder::Builder;
use parking_lot::RwLock;

use crate::{
    amplitude::AmplitudeBuilder,
    dataset::DatasetError,
    prelude::{
        Amplitude, CMatrix64, CScalar64, CVector64, Dataset, Entry, IntoAmplitude, Matrix64,
        Momenta64, Momentum64, ParMap, Scalar64, Vector64,
    },
};

pub type ResolveToScalar64 = dyn Fn(&Entry) -> Scalar64 + Send + Sync;
pub type ResolveToCScalar64 = dyn Fn(&Entry) -> CScalar64 + Send + Sync;
pub type ResolveToVector64 = dyn Fn(&Entry) -> Vector64 + Send + Sync;
pub type ResolveToCVector64 = dyn Fn(&Entry) -> CVector64 + Send + Sync;
pub type ResolveToMatrix64 = dyn Fn(&Entry) -> Matrix64 + Send + Sync;
pub type ResolveToCMatrix64 = dyn Fn(&Entry) -> CMatrix64 + Send + Sync;
pub type ResolveToMomentum64 = dyn Fn(&Entry) -> Momentum64 + Send + Sync;
pub type ResolveToMomenta64 = dyn Fn(&Entry) -> Momenta64 + Send + Sync;

pub trait IntoVariable {
    fn into_variable(self) -> Variable;
}

#[derive(Clone, Variantly)]
pub enum Variable {
    #[variantly(rename = "scalar")]
    Scalar(ScalarVariable),
    #[variantly(rename = "cscalar")]
    CScalar(CScalarVariable),
    #[variantly(rename = "vector")]
    Vector(VectorVariable),
    #[variantly(rename = "cvector")]
    CVector(CVectorVariable),
    #[variantly(rename = "matrix")]
    Matrix(MatrixVariable),
    #[variantly(rename = "cmatrix")]
    CMatrix(CMatrixVariable),
    #[variantly(rename = "momentum")]
    Momentum(MomentumVariable),
    #[variantly(rename = "momenta")]
    Momenta(MomentaVariable),
}

pub trait Resolve {
    fn resolve(&self, dataset: &mut Dataset, prunable: bool) -> Result<(), DatasetError>;
    fn resolve_par(&self, dataset: &mut Dataset, prunable: bool) -> Result<(), DatasetError>;
}

#[derive(Builder, Clone)]
pub struct ScalarVariable {
    #[builder(setter(into))]
    pub name: String,
    #[builder(setter(custom))]
    function: Arc<RwLock<ResolveToScalar64>>,
    #[builder(setter(custom), default)]
    dependencies: Option<Arc<RwLock<Vec<Variable>>>>,
}

impl ScalarVariableBuilder {
    pub fn function<F>(&mut self, f: F) -> &mut Self
    where
        F: 'static + Fn(&Entry) -> Scalar64 + Send + Sync,
    {
        self.function = Some(Arc::new(RwLock::new(f)));
        self
    }

    #[anyinput]
    pub fn dependencies(&mut self, variables: AnyIter<Variable>) -> &mut Self {
        self.dependencies = Some(Some(Arc::new(RwLock::new(
            variables.into_iter().collect::<Vec<_>>(),
        ))));
        self
    }
}

impl IntoAmplitude for ScalarVariable {
    fn into_amplitude(self) -> Amplitude {
        AmplitudeBuilder::default()
            .name(self.name.clone())
            .dependencies([Variable::Scalar(self.clone())])
            .function(move |_pars: &ParMap, vars: &Entry| {
                Ok(vars.scalar(self.name.clone()).unwrap().into())
            })
            .build()
            .unwrap()
    }
}

impl Resolve for ScalarVariable {
    fn resolve(&self, dataset: &mut Dataset, prunable: bool) -> Result<(), DatasetError> {
        if dataset.contains_field(self.name.clone()) {
            return Ok(());
        }
        if let Some(deps) = self.dependencies.clone() {
            deps.read().iter().for_each(|dep| {
                match dep {
                    Variable::Scalar(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CScalar(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Vector(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CVector(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Matrix(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CMatrix(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Momentum(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Momenta(var) => var.resolve(dataset, true).unwrap(),
                };
            })
        }
        let field: Vec<Scalar64> = dataset
            .entries
            .iter()
            .map(|entry| self.function.read()(&entry.read()))
            .collect();
        dataset.add_scalar_field(self.name.clone(), field, prunable)
    }
    fn resolve_par(&self, dataset: &mut Dataset, prunable: bool) -> Result<(), DatasetError> {
        if dataset.contains_field(self.name.clone()) {
            return Ok(());
        }
        if let Some(deps) = self.dependencies.clone() {
            deps.read().iter().for_each(|dep| {
                match dep {
                    Variable::Scalar(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CScalar(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Vector(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CVector(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Matrix(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CMatrix(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Momentum(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Momenta(var) => var.resolve_par(dataset, true).unwrap(),
                };
            })
        }
        let field: Vec<Scalar64> = dataset
            .entries
            .par_iter()
            .map(|entry| self.function.read()(&entry.read()))
            .collect();
        dataset.add_scalar_field_par(self.name.clone(), field, prunable)
    }
}

#[derive(Builder, Clone)]
pub struct CScalarVariable {
    #[builder(setter(into))]
    pub name: String,
    #[builder(setter(custom))]
    function: Arc<RwLock<ResolveToCScalar64>>,
    #[builder(setter(custom), default)]
    dependencies: Option<Arc<RwLock<Vec<Variable>>>>,
}

impl CScalarVariableBuilder {
    pub fn function<F>(&mut self, f: F) -> &mut Self
    where
        F: 'static + Fn(&Entry) -> CScalar64 + Send + Sync,
    {
        self.function = Some(Arc::new(RwLock::new(f)));
        self
    }

    #[anyinput]
    pub fn dependencies(&mut self, variables: AnyIter<Variable>) -> &mut Self {
        self.dependencies = Some(Some(Arc::new(RwLock::new(
            variables.into_iter().collect::<Vec<_>>(),
        ))));
        self
    }
}

impl IntoAmplitude for CScalarVariable {
    fn into_amplitude(self) -> Amplitude {
        AmplitudeBuilder::default()
            .name(self.clone().name)
            .dependencies([Variable::CScalar(self.clone())])
            .function(move |_pars: &ParMap, vars: &Entry| {
                Ok(vars.cscalar(self.clone().name).unwrap())
            })
            .build()
            .unwrap()
    }
}

impl Resolve for CScalarVariable {
    fn resolve(&self, dataset: &mut Dataset, prunable: bool) -> Result<(), DatasetError> {
        if dataset.contains_field(self.name.clone()) {
            return Ok(());
        }
        if let Some(deps) = self.dependencies.clone() {
            deps.read().iter().for_each(|dep| {
                match dep {
                    Variable::Scalar(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CScalar(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Vector(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CVector(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Matrix(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CMatrix(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Momentum(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Momenta(var) => var.resolve(dataset, true).unwrap(),
                };
            })
        }
        let field: Vec<CScalar64> = dataset
            .entries
            .iter()
            .map(|entry| self.function.read()(&entry.read()))
            .collect();
        dataset.add_cscalar_field(self.name.clone(), field, prunable)
    }
    fn resolve_par(&self, dataset: &mut Dataset, prunable: bool) -> Result<(), DatasetError> {
        if dataset.contains_field(self.name.clone()) {
            return Ok(());
        }
        if let Some(deps) = self.dependencies.clone() {
            deps.read().iter().for_each(|dep| {
                match dep {
                    Variable::Scalar(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CScalar(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Vector(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CVector(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Matrix(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CMatrix(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Momentum(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Momenta(var) => var.resolve_par(dataset, true).unwrap(),
                };
            })
        }
        let field: Vec<CScalar64> = dataset
            .entries
            .par_iter()
            .map(|entry| self.function.read()(&entry.read()))
            .collect();
        dataset.add_cscalar_field_par(self.name.clone(), field, prunable)
    }
}

#[derive(Builder, Clone)]
pub struct VectorVariable {
    #[builder(setter(into))]
    pub name: String,
    #[builder(setter(custom))]
    function: Arc<RwLock<ResolveToVector64>>,
    #[builder(setter(custom), default)]
    dependencies: Option<Arc<RwLock<Vec<Variable>>>>,
}

impl VectorVariableBuilder {
    pub fn function<F>(&mut self, f: F) -> &mut Self
    where
        F: 'static + Fn(&Entry) -> Vector64 + Send + Sync,
    {
        self.function = Some(Arc::new(RwLock::new(f)));
        self
    }

    #[anyinput]
    pub fn dependencies(&mut self, variables: AnyIter<Variable>) -> &mut Self {
        self.dependencies = Some(Some(Arc::new(RwLock::new(
            variables.into_iter().collect::<Vec<_>>(),
        ))));
        self
    }
}

impl Resolve for VectorVariable {
    fn resolve(&self, dataset: &mut Dataset, prunable: bool) -> Result<(), DatasetError> {
        if dataset.contains_field(self.name.clone()) {
            return Ok(());
        }
        if let Some(deps) = self.dependencies.clone() {
            deps.read().iter().for_each(|dep| {
                match dep {
                    Variable::Scalar(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CScalar(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Vector(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CVector(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Matrix(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CMatrix(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Momentum(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Momenta(var) => var.resolve(dataset, true).unwrap(),
                };
            })
        }
        let field: Vec<Vector64> = dataset
            .entries
            .iter()
            .map(|entry| self.function.read()(&entry.read()))
            .collect();
        dataset.add_vector_field(self.name.clone(), field, prunable)
    }
    fn resolve_par(&self, dataset: &mut Dataset, prunable: bool) -> Result<(), DatasetError> {
        if dataset.contains_field(self.name.clone()) {
            return Ok(());
        }
        if let Some(deps) = self.dependencies.clone() {
            deps.read().iter().for_each(|dep| {
                match dep {
                    Variable::Scalar(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CScalar(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Vector(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CVector(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Matrix(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CMatrix(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Momentum(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Momenta(var) => var.resolve_par(dataset, true).unwrap(),
                };
            })
        }
        let field: Vec<Vector64> = dataset
            .entries
            .par_iter()
            .map(|entry| self.function.read()(&entry.read()))
            .collect();
        dataset.add_vector_field_par(self.name.clone(), field, prunable)
    }
}

#[derive(Builder, Clone)]
pub struct CVectorVariable {
    #[builder(setter(into))]
    pub name: String,
    #[builder(setter(custom))]
    function: Arc<RwLock<ResolveToCVector64>>,
    #[builder(setter(custom), default)]
    dependencies: Option<Arc<RwLock<Vec<Variable>>>>,
}

impl CVectorVariableBuilder {
    pub fn function<F>(&mut self, f: F) -> &mut Self
    where
        F: 'static + Fn(&Entry) -> CVector64 + Send + Sync,
    {
        self.function = Some(Arc::new(RwLock::new(f)));
        self
    }

    #[anyinput]
    pub fn dependencies(&mut self, variables: AnyIter<Variable>) -> &mut Self {
        self.dependencies = Some(Some(Arc::new(RwLock::new(
            variables.into_iter().collect::<Vec<_>>(),
        ))));
        self
    }
}

impl Resolve for CVectorVariable {
    fn resolve(&self, dataset: &mut Dataset, prunable: bool) -> Result<(), DatasetError> {
        if dataset.contains_field(self.name.clone()) {
            return Ok(());
        }
        if let Some(deps) = self.dependencies.clone() {
            deps.read().iter().for_each(|dep| {
                match dep {
                    Variable::Scalar(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CScalar(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Vector(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CVector(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Matrix(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CMatrix(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Momentum(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Momenta(var) => var.resolve(dataset, true).unwrap(),
                };
            })
        }
        let field: Vec<CVector64> = dataset
            .entries
            .iter()
            .map(|entry| self.function.read()(&entry.read()))
            .collect();
        dataset.add_cvector_field(self.name.clone(), field, prunable)
    }
    fn resolve_par(&self, dataset: &mut Dataset, prunable: bool) -> Result<(), DatasetError> {
        if dataset.contains_field(self.name.clone()) {
            return Ok(());
        }
        if let Some(deps) = self.dependencies.clone() {
            deps.read().iter().for_each(|dep| {
                match dep {
                    Variable::Scalar(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CScalar(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Vector(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CVector(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Matrix(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CMatrix(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Momentum(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Momenta(var) => var.resolve_par(dataset, true).unwrap(),
                };
            })
        }
        let field: Vec<CVector64> = dataset
            .entries
            .par_iter()
            .map(|entry| self.function.read()(&entry.read()))
            .collect();
        dataset.add_cvector_field_par(self.name.clone(), field, prunable)
    }
}

#[derive(Builder, Clone)]
pub struct MatrixVariable {
    #[builder(setter(into))]
    pub name: String,
    #[builder(setter(custom))]
    function: Arc<RwLock<ResolveToMatrix64>>,
    #[builder(setter(custom), default)]
    dependencies: Option<Arc<RwLock<Vec<Variable>>>>,
}

impl MatrixVariableBuilder {
    pub fn function<F>(&mut self, f: F) -> &mut Self
    where
        F: 'static + Fn(&Entry) -> Matrix64 + Send + Sync,
    {
        self.function = Some(Arc::new(RwLock::new(f)));
        self
    }

    #[anyinput]
    pub fn dependencies(&mut self, variables: AnyIter<Variable>) -> &mut Self {
        self.dependencies = Some(Some(Arc::new(RwLock::new(
            variables.into_iter().collect::<Vec<_>>(),
        ))));
        self
    }
}

impl Resolve for MatrixVariable {
    fn resolve(&self, dataset: &mut Dataset, prunable: bool) -> Result<(), DatasetError> {
        if dataset.contains_field(self.name.clone()) {
            return Ok(());
        }
        if let Some(deps) = self.dependencies.clone() {
            deps.read().iter().for_each(|dep| {
                match dep {
                    Variable::Scalar(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CScalar(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Vector(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CVector(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Matrix(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CMatrix(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Momentum(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Momenta(var) => var.resolve(dataset, true).unwrap(),
                };
            })
        }
        let field: Vec<Matrix64> = dataset
            .entries
            .iter()
            .map(|entry| self.function.read()(&entry.read()))
            .collect();
        dataset.add_matrix_field(self.name.clone(), field, prunable)
    }
    fn resolve_par(&self, dataset: &mut Dataset, prunable: bool) -> Result<(), DatasetError> {
        if dataset.contains_field(self.name.clone()) {
            return Ok(());
        }
        if let Some(deps) = self.dependencies.clone() {
            deps.read().iter().for_each(|dep| {
                match dep {
                    Variable::Scalar(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CScalar(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Vector(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CVector(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Matrix(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CMatrix(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Momentum(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Momenta(var) => var.resolve_par(dataset, true).unwrap(),
                };
            })
        }
        let field: Vec<Matrix64> = dataset
            .entries
            .par_iter()
            .map(|entry| self.function.read()(&entry.read()))
            .collect();
        dataset.add_matrix_field_par(self.name.clone(), field, prunable)
    }
}

#[derive(Builder, Clone)]
pub struct CMatrixVariable {
    #[builder(setter(into))]
    pub name: String,
    #[builder(setter(custom))]
    function: Arc<RwLock<ResolveToCMatrix64>>,
    #[builder(setter(custom), default)]
    dependencies: Option<Arc<RwLock<Vec<Variable>>>>,
}

impl CMatrixVariableBuilder {
    pub fn function<F>(&mut self, f: F) -> &mut Self
    where
        F: 'static + Fn(&Entry) -> CMatrix64 + Send + Sync,
    {
        self.function = Some(Arc::new(RwLock::new(f)));
        self
    }

    #[anyinput]
    pub fn dependencies(&mut self, variables: AnyIter<Variable>) -> &mut Self {
        self.dependencies = Some(Some(Arc::new(RwLock::new(
            variables.into_iter().collect::<Vec<_>>(),
        ))));
        self
    }
}

impl Resolve for CMatrixVariable {
    fn resolve(&self, dataset: &mut Dataset, prunable: bool) -> Result<(), DatasetError> {
        if dataset.contains_field(self.name.clone()) {
            return Ok(());
        }
        if let Some(deps) = self.dependencies.clone() {
            deps.read().iter().for_each(|dep| {
                match dep {
                    Variable::Scalar(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CScalar(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Vector(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CVector(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Matrix(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CMatrix(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Momentum(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Momenta(var) => var.resolve(dataset, true).unwrap(),
                };
            })
        }
        let field: Vec<CMatrix64> = dataset
            .entries
            .iter()
            .map(|entry| self.function.read()(&entry.read()))
            .collect();
        dataset.add_cmatrix_field(self.name.clone(), field, prunable)
    }
    fn resolve_par(&self, dataset: &mut Dataset, prunable: bool) -> Result<(), DatasetError> {
        if dataset.contains_field(self.name.clone()) {
            return Ok(());
        }
        if let Some(deps) = self.dependencies.clone() {
            deps.read().iter().for_each(|dep| {
                match dep {
                    Variable::Scalar(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CScalar(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Vector(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CVector(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Matrix(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CMatrix(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Momentum(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Momenta(var) => var.resolve_par(dataset, true).unwrap(),
                };
            })
        }
        let field: Vec<CMatrix64> = dataset
            .entries
            .par_iter()
            .map(|entry| self.function.read()(&entry.read()))
            .collect();
        dataset.add_cmatrix_field_par(self.name.clone(), field, prunable)
    }
}

#[derive(Builder, Clone)]
pub struct MomentumVariable {
    #[builder(setter(into))]
    pub name: String,
    #[builder(setter(custom))]
    function: Arc<RwLock<ResolveToMomentum64>>,
    #[builder(setter(custom), default)]
    dependencies: Option<Arc<RwLock<Vec<Variable>>>>,
}

impl MomentumVariableBuilder {
    pub fn function<F>(&mut self, f: F) -> &mut Self
    where
        F: 'static + Fn(&Entry) -> Momentum64 + Send + Sync,
    {
        self.function = Some(Arc::new(RwLock::new(f)));
        self
    }

    #[anyinput]
    pub fn dependencies(&mut self, variables: AnyIter<Variable>) -> &mut Self {
        self.dependencies = Some(Some(Arc::new(RwLock::new(
            variables.into_iter().collect::<Vec<_>>(),
        ))));
        self
    }
}

impl Resolve for MomentumVariable {
    fn resolve(&self, dataset: &mut Dataset, prunable: bool) -> Result<(), DatasetError> {
        if dataset.contains_field(self.name.clone()) {
            return Ok(());
        }
        if let Some(deps) = self.dependencies.clone() {
            deps.read().iter().for_each(|dep| {
                match dep {
                    Variable::Scalar(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CScalar(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Vector(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CVector(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Matrix(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CMatrix(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Momentum(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Momenta(var) => var.resolve(dataset, true).unwrap(),
                };
            })
        }
        let field: Vec<Momentum64> = dataset
            .entries
            .iter()
            .map(|entry| self.function.read()(&entry.read()))
            .collect();
        dataset.add_momentum_field(self.name.clone(), field, prunable)
    }
    fn resolve_par(&self, dataset: &mut Dataset, prunable: bool) -> Result<(), DatasetError> {
        if dataset.contains_field(self.name.clone()) {
            return Ok(());
        }
        if let Some(deps) = self.dependencies.clone() {
            deps.read().iter().for_each(|dep| {
                match dep {
                    Variable::Scalar(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CScalar(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Vector(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CVector(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Matrix(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CMatrix(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Momentum(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Momenta(var) => var.resolve_par(dataset, true).unwrap(),
                };
            })
        }
        let field: Vec<Momentum64> = dataset
            .entries
            .par_iter()
            .map(|entry| self.function.read()(&entry.read()))
            .collect();
        dataset.add_momentum_field_par(self.name.clone(), field, prunable)
    }
}
#[derive(Builder, Clone)]
pub struct MomentaVariable {
    #[builder(setter(into))]
    pub name: String,
    #[builder(setter(custom))]
    function: Arc<RwLock<ResolveToMomenta64>>,
    #[builder(setter(custom), default)]
    dependencies: Option<Arc<RwLock<Vec<Variable>>>>,
}

impl MomentaVariableBuilder {
    pub fn function<F>(&mut self, f: F) -> &mut Self
    where
        F: 'static + Fn(&Entry) -> Momenta64 + Send + Sync,
    {
        self.function = Some(Arc::new(RwLock::new(f)));
        self
    }

    #[anyinput]
    pub fn dependencies(&mut self, variables: AnyIter<Variable>) -> &mut Self {
        self.dependencies = Some(Some(Arc::new(RwLock::new(
            variables.into_iter().collect::<Vec<_>>(),
        ))));
        self
    }
}

impl Resolve for MomentaVariable {
    fn resolve(&self, dataset: &mut Dataset, prunable: bool) -> Result<(), DatasetError> {
        if dataset.contains_field(self.name.clone()) {
            return Ok(());
        }
        if let Some(deps) = self.dependencies.clone() {
            deps.read().iter().for_each(|dep| {
                match dep {
                    Variable::Scalar(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CScalar(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Vector(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CVector(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Matrix(var) => var.resolve(dataset, true).unwrap(),
                    Variable::CMatrix(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Momentum(var) => var.resolve(dataset, true).unwrap(),
                    Variable::Momenta(var) => var.resolve(dataset, true).unwrap(),
                };
            })
        }
        let field: Vec<Momenta64> = dataset
            .entries
            .iter()
            .map(|entry| self.function.read()(&entry.read()))
            .collect();
        dataset.add_momenta_field(self.name.clone(), field, prunable)
    }
    fn resolve_par(&self, dataset: &mut Dataset, prunable: bool) -> Result<(), DatasetError> {
        if dataset.contains_field(self.name.clone()) {
            return Ok(());
        }
        if let Some(deps) = self.dependencies.clone() {
            deps.read().iter().for_each(|dep| {
                match dep {
                    Variable::Scalar(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CScalar(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Vector(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CVector(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Matrix(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::CMatrix(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Momentum(var) => var.resolve_par(dataset, true).unwrap(),
                    Variable::Momenta(var) => var.resolve_par(dataset, true).unwrap(),
                };
            })
        }
        let field: Vec<Momenta64> = dataset
            .entries
            .par_iter()
            .map(|entry| self.function.read()(&entry.read()))
            .collect();
        dataset.add_momenta_field_par(self.name.clone(), field, prunable)
    }
}
