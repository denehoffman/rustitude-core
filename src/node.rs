use num_complex::{Complex64, ComplexFloat};
use rustc_hash::FxHashMap as HashMap;

use anyinput::anyinput;
use rayon::prelude::*;
use thiserror::Error;

use crate::prelude::{CScalar64, Dataset};

#[derive(Error, Debug)]
pub enum ParameterError {
    #[error("Parameter not found: {parameter_name}")]
    ParameterNotFound { parameter_name: String },
}

pub trait Dependent {
    fn dependencies(&self) -> Vec<&dyn Resolvable> {
        vec![]
    }
}
pub trait Resolvable: Dependent {
    #[allow(unused_variables)]
    fn compute(&self, ds: &mut Dataset) {}
    fn resolve(&self, ds: &mut Dataset) {
        for resource in self.dependencies() {
            resource.resolve(ds)
        }
        self.compute(ds);
    }
}

pub trait Parameterized: Resolvable {
    fn get_external_par_name(&self, internal_par_name: &str) -> Option<&String>;
    fn get_par_by_name(
        &self,
        internal_par_name: &str,
        pars: &HashMap<String, f64>,
    ) -> Result<f64, ParameterError> {
        if let Some(external_name) = self.get_external_par_name(internal_par_name) {
            Ok(*pars.get(external_name).unwrap()) // TODO: param was mapped but not passed in
        } else {
            Err(ParameterError::ParameterNotFound {
                parameter_name: internal_par_name.to_string(), // TODO: param was passed in but not
                                                               // mapped
            })
        }
    }
}

pub trait Node: Resolvable {
    fn eval(&self, ds: &Dataset, pars: &HashMap<String, f64>) -> Vec<CScalar64>;
    fn abs(&self) -> AbsNode<Self>
    where
        Self: Sized + Clone,
    {
        AbsNode { a: self.clone() }
    }
    fn norm_sqr(&self) -> NormSqrNode<Self>
    where
        Self: Sized + Clone,
    {
        NormSqrNode { a: self.clone() }
    }
    fn re(&self) -> RealNode<Self>
    where
        Self: Sized + Clone,
    {
        RealNode { a: self.clone() }
    }
    fn real(&self) -> RealNode<Self>
    where
        Self: Sized + Clone,
    {
        RealNode { a: self.clone() }
    }
    fn im(&self) -> ImagNode<Self>
    where
        Self: Sized + Clone,
    {
        ImagNode { a: self.clone() }
    }
    fn imag(&self) -> ImagNode<Self>
    where
        Self: Sized + Clone,
    {
        ImagNode { a: self.clone() }
    }
    fn neg(&self) -> NegNode<Self>
    where
        Self: Sized + Clone,
    {
        NegNode { a: self.clone() }
    }
    fn add<U>(&self, other: &U) -> AddNode<Self, U>
    where
        Self: Sized + Clone,
        U: Node + Clone,
    {
        AddNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn sub<U>(&self, other: &U) -> SubNode<Self, U>
    where
        Self: Sized + Clone,
        U: Node + Clone,
    {
        SubNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn mul<U>(&self, other: &U) -> MulNode<Self, U>
    where
        Self: Sized + Clone,
        U: Node + Clone,
    {
        MulNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn div<U>(&self, other: &U) -> DivNode<Self, U>
    where
        Self: Sized + Clone,
        U: Node + Clone,
    {
        DivNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn pow<U>(&self, other: &U) -> PowNode<Self, U>
    where
        Self: Sized + Clone,
        U: Node + Clone,
    {
        PowNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
}

#[derive(Clone)]
pub struct ParameterNode(HashMap<String, String>);
impl ParameterNode {
    #[anyinput]
    pub fn new(name: AnyString) -> Self {
        let mut p = ParameterNode(HashMap::default());
        p.0.insert("parameter".to_string(), name.to_string());
        p
    }
}
impl Dependent for ParameterNode {}
impl Resolvable for ParameterNode {}
impl Parameterized for ParameterNode {
    fn get_external_par_name(&self, internal_par_name: &str) -> Option<&String> {
        self.0.get(internal_par_name)
    }
}
impl Node for ParameterNode {
    fn eval(&self, ds: &Dataset, pars: &HashMap<String, f64>) -> Vec<CScalar64> {
        let p = CScalar64::new(self.get_par_by_name("parameter", pars).unwrap(), 0.0);
        vec![p; ds.len()]
    }
}

#[derive(Clone)]
pub struct ComplexParameterNode(HashMap<String, String>);
impl ComplexParameterNode {
    #[anyinput]
    pub fn new(name_re: AnyString, name_im: AnyString) -> Self {
        let mut p = ComplexParameterNode(HashMap::default());
        p.0.insert("parameter (re)".to_string(), name_re.to_string());
        p.0.insert("parameter (im)".to_string(), name_im.to_string());
        p
    }
}
impl Dependent for ComplexParameterNode {}
impl Resolvable for ComplexParameterNode {}
impl Parameterized for ComplexParameterNode {
    fn get_external_par_name(&self, internal_par_name: &str) -> Option<&String> {
        self.0.get(internal_par_name)
    }
}
impl Node for ComplexParameterNode {
    fn eval(&self, ds: &Dataset, pars: &HashMap<String, f64>) -> Vec<CScalar64> {
        let p_re = self.get_par_by_name("parameter (re)", pars).unwrap();
        let p_im = self.get_par_by_name("parameter (im)", pars).unwrap();
        let p = CScalar64::new(p_re, p_im);
        vec![p; ds.len()]
    }
}

#[derive(Clone)]
pub struct ConstantNode(Complex64);
impl Dependent for ConstantNode {}
impl Resolvable for ConstantNode {}
impl Node for ConstantNode {
    fn eval(&self, ds: &Dataset, _pars: &HashMap<String, f64>) -> Vec<CScalar64> {
        vec![self.0; ds.len()]
    }
}
impl From<Complex64> for ConstantNode {
    fn from(value: Complex64) -> Self {
        ConstantNode(value)
    }
}
impl From<f64> for ConstantNode {
    fn from(value: f64) -> Self {
        ConstantNode(value.into())
    }
}

macro_rules! unary_op {
    ($name:ident, $func:expr) => {
        #[derive(Clone)]
        pub struct $name<T>
        where
            T: Node,
        {
            a: T,
        }

        impl<T> Dependent for $name<T>
        where
            T: Node,
        {
            fn dependencies(&self) -> Vec<&dyn Resolvable> {
                vec![&self.a]
            }
        }
        impl<T> Resolvable for $name<T> where T: Node {}
        impl<T> Node for $name<T>
        where
            T: Node,
        {
            fn eval(&self, ds: &Dataset, pars: &HashMap<String, f64>) -> Vec<CScalar64> {
                let eval_a = self.a.eval(ds, pars);

                eval_a.into_par_iter().map($func).collect()
            }
        }
    };
}

macro_rules! binary_op {
    ($name:ident, $func:expr) => {
        #[derive(Clone)]
        pub struct $name<T, U>
        where
            T: Node,
            U: Node,
        {
            a: T,
            b: U,
        }

        impl<T, U> Dependent for $name<T, U>
        where
            T: Node,
            U: Node,
        {
            fn dependencies(&self) -> Vec<&dyn Resolvable> {
                vec![&self.a, &self.b]
            }
        }
        impl<T, U> Resolvable for $name<T, U>
        where
            T: Node,
            U: Node,
        {
        }
        impl<T, U> Node for $name<T, U>
        where
            T: Node,
            U: Node,
        {
            fn eval(&self, ds: &Dataset, pars: &HashMap<String, f64>) -> Vec<CScalar64> {
                let eval_a = self.a.eval(ds, pars);
                let eval_b = self.b.eval(ds, pars);

                (eval_a, eval_b).into_par_iter().map($func).collect()
            }
        }
    };
}

unary_op!(AbsNode, |a| CScalar64::abs(a).into());
unary_op!(NormSqrNode, |a| { CScalar64::norm_sqr(&a).into() });
unary_op!(RealNode, |a| CScalar64::re(a).into());
unary_op!(ImagNode, |a| CScalar64::im(a).into());
unary_op!(NegNode, |a| -a);
binary_op!(AddNode, |(a, b)| a + b);
binary_op!(SubNode, |(a, b)| a - b);
binary_op!(MulNode, |(a, b)| a * b);
binary_op!(DivNode, |(a, b)| a / b);
binary_op!(PowNode, |(a, b)| { CScalar64::powc(a, b) });
