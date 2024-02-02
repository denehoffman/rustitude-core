use num_complex::ComplexFloat;
use std::collections::HashMap;

use anyinput::anyinput;
use rayon::prelude::*;
use thiserror::Error;

use crate::prelude::{CScalar64, Dataset, Scalar64};

#[derive(Error, Debug)]
pub enum ParameterError {
    #[error("Parameter not found: {parameter_name}")]
    ParameterNotFound { parameter_name: String },
}

pub trait Dependent {
    fn dependencies(&self) -> Vec<&dyn Resolvable>;
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

pub trait SNode: Resolvable {
    fn eval(&self, ds: &Dataset, pars: &HashMap<String, f64>) -> Vec<Scalar64>;
    fn abs(&self) -> AbsSSNode<Self>
    where
        Self: Sized + Clone,
    {
        AbsSSNode { a: self.clone() }
    }
    fn neg(&self) -> NegSSNode<Self>
    where
        Self: Sized + Clone,
    {
        NegSSNode { a: self.clone() }
    }
    fn add<U>(&self, other: &U) -> AddSSSNode<Self, U>
    where
        Self: Sized + Clone,
        U: SNode + Clone,
    {
        AddSSSNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn sub<U>(&self, other: &U) -> SubSSSNode<Self, U>
    where
        Self: Sized + Clone,
        U: SNode + Clone,
    {
        SubSSSNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn mul<U>(&self, other: &U) -> MulSSSNode<Self, U>
    where
        Self: Sized + Clone,
        U: SNode + Clone,
    {
        MulSSSNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn div<U>(&self, other: &U) -> DivSSSNode<Self, U>
    where
        Self: Sized + Clone,
        U: SNode + Clone,
    {
        DivSSSNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn pow<U>(&self, other: &U) -> PowSSSNode<Self, U>
    where
        Self: Sized + Clone,
        U: SNode + Clone,
    {
        PowSSSNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn add_complex<U>(&self, other: &U) -> AddSCCNode<Self, U>
    where
        Self: Sized + Clone,
        U: CNode + Clone,
    {
        AddSCCNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn sub_complex<U>(&self, other: &U) -> SubSCCNode<Self, U>
    where
        Self: Sized + Clone,
        U: CNode + Clone,
    {
        SubSCCNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn mul_complex<U>(&self, other: &U) -> MulSCCNode<Self, U>
    where
        Self: Sized + Clone,
        U: CNode + Clone,
    {
        MulSCCNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn div_complex<U>(&self, other: &U) -> DivSCCNode<Self, U>
    where
        Self: Sized + Clone,
        U: CNode + Clone,
    {
        DivSCCNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn pow_complex<U>(&self, other: &U) -> PowSCCNode<Self, U>
    where
        Self: Sized + Clone,
        U: CNode + Clone,
    {
        PowSCCNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
}
pub trait CNode: Resolvable {
    fn eval(&self, ds: &Dataset, pars: &HashMap<String, f64>) -> Vec<CScalar64>;
    fn abs(&self) -> AbsCSNode<Self>
    where
        Self: Sized + Clone,
    {
        AbsCSNode { a: self.clone() }
    }
    fn norm_sqr(&self) -> NormSqrCSNode<Self>
    where
        Self: Sized + Clone,
    {
        NormSqrCSNode { a: self.clone() }
    }
    fn re(&self) -> RealCSNode<Self>
    where
        Self: Sized + Clone,
    {
        RealCSNode { a: self.clone() }
    }
    fn real(&self) -> RealCSNode<Self>
    where
        Self: Sized + Clone,
    {
        RealCSNode { a: self.clone() }
    }
    fn im(&self) -> ImagCSNode<Self>
    where
        Self: Sized + Clone,
    {
        ImagCSNode { a: self.clone() }
    }
    fn imag(&self) -> ImagCSNode<Self>
    where
        Self: Sized + Clone,
    {
        ImagCSNode { a: self.clone() }
    }
    fn neg(&self) -> NegCCNode<Self>
    where
        Self: Sized + Clone,
    {
        NegCCNode { a: self.clone() }
    }
    fn add<U>(&self, other: &U) -> AddCCCNode<Self, U>
    where
        Self: Sized + Clone,
        U: CNode + Clone,
    {
        AddCCCNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn sub<U>(&self, other: &U) -> SubCCCNode<Self, U>
    where
        Self: Sized + Clone,
        U: CNode + Clone,
    {
        SubCCCNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn mul<U>(&self, other: &U) -> MulCCCNode<Self, U>
    where
        Self: Sized + Clone,
        U: CNode + Clone,
    {
        MulCCCNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn div<U>(&self, other: &U) -> DivCCCNode<Self, U>
    where
        Self: Sized + Clone,
        U: CNode + Clone,
    {
        DivCCCNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn pow<U>(&self, other: &U) -> PowCCCNode<Self, U>
    where
        Self: Sized + Clone,
        U: CNode + Clone,
    {
        PowCCCNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn add_scalar<U>(&self, other: &U) -> AddCSCNode<Self, U>
    where
        Self: Sized + Clone,
        U: SNode + Clone,
    {
        AddCSCNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn sub_scalar<U>(&self, other: &U) -> SubCSCNode<Self, U>
    where
        Self: Sized + Clone,
        U: SNode + Clone,
    {
        SubCSCNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn mul_scalar<U>(&self, other: &U) -> MulCSCNode<Self, U>
    where
        Self: Sized + Clone,
        U: SNode + Clone,
    {
        MulCSCNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn div_scalar<U>(&self, other: &U) -> DivCSCNode<Self, U>
    where
        Self: Sized + Clone,
        U: SNode + Clone,
    {
        DivCSCNode {
            a: self.clone(),
            b: other.clone(),
        }
    }
    fn pow_scalar<U>(&self, other: &U) -> PowCSCNode<Self, U>
    where
        Self: Sized + Clone,
        U: SNode + Clone,
    {
        PowCSCNode {
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
        return p;
    }
}
impl Dependent for ParameterNode {
    fn dependencies(&self) -> Vec<&dyn Resolvable> {
        vec![]
    }
}
impl Resolvable for ParameterNode {}
impl Parameterized for ParameterNode {
    fn get_external_par_name(&self, internal_par_name: &str) -> Option<&String> {
        self.0.get(internal_par_name)
    }
}
impl SNode for ParameterNode {
    fn eval(&self, ds: &Dataset, pars: &HashMap<String, f64>) -> Vec<Scalar64> {
        let p = self.get_par_by_name("parameter", pars).unwrap();
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
        return p;
    }
}
impl Dependent for ComplexParameterNode {
    fn dependencies(&self) -> Vec<&dyn Resolvable> {
        vec![]
    }
}
impl Resolvable for ComplexParameterNode {}
impl Parameterized for ComplexParameterNode {
    fn get_external_par_name(&self, internal_par_name: &str) -> Option<&String> {
        self.0.get(internal_par_name)
    }
}
impl CNode for ComplexParameterNode {
    fn eval(&self, ds: &Dataset, pars: &HashMap<String, f64>) -> Vec<CScalar64> {
        let p_re = self.get_par_by_name("parameter (re)", pars).unwrap();
        let p_im = self.get_par_by_name("parameter (im)", pars).unwrap();
        let p = CScalar64::new(p_re, p_im);
        vec![p; ds.len()]
    }
}

macro_rules! unary_op {
    ($name:ident, $a:ident, $output:ident, $outtype:ty, $func:expr) => {
        #[derive(Clone)]
        pub struct $name<T>
        where
            T: $a,
        {
            a: T,
        }

        impl<T> Dependent for $name<T>
        where
            T: $a,
        {
            fn dependencies(&self) -> Vec<&dyn Resolvable> {
                vec![&self.a]
            }
        }
        impl<T> Resolvable for $name<T> where T: $a {}
        impl<T> $output for $name<T>
        where
            T: $a,
        {
            fn eval(&self, ds: &Dataset, pars: &HashMap<String, f64>) -> Vec<$outtype> {
                let eval_a = self.a.eval(ds, pars);

                eval_a.into_par_iter().map($func).collect()
            }
        }
    };
}

macro_rules! binary_op {
    ($name:ident, $a:ident, $b:ident, $output:ident, $outtype:ty, $func:expr) => {
        #[derive(Clone)]
        pub struct $name<T, U>
        where
            T: $a,
            U: $b,
        {
            a: T,
            b: U,
        }

        impl<T, U> Dependent for $name<T, U>
        where
            T: $a,
            U: $b,
        {
            fn dependencies(&self) -> Vec<&dyn Resolvable> {
                vec![&self.a, &self.b]
            }
        }
        impl<T, U> Resolvable for $name<T, U>
        where
            T: $a,
            U: $b,
        {
        }
        impl<T, U> $output for $name<T, U>
        where
            T: $a,
            U: $b,
        {
            fn eval(&self, ds: &Dataset, pars: &HashMap<String, f64>) -> Vec<$outtype> {
                let eval_a = self.a.eval(ds, pars);
                let eval_b = self.b.eval(ds, pars);

                (eval_a, eval_b).into_par_iter().map($func).collect()
            }
        }
    };
}

unary_op!(AbsSSNode, SNode, SNode, Scalar64, |a| f64::abs(a));
unary_op!(NegSSNode, SNode, SNode, Scalar64, |a| -a);
binary_op!(AddSSSNode, SNode, SNode, SNode, Scalar64, |(a, b)| a + b);
binary_op!(SubSSSNode, SNode, SNode, SNode, Scalar64, |(a, b)| a - b);
binary_op!(MulSSSNode, SNode, SNode, SNode, Scalar64, |(a, b)| a * b);
binary_op!(DivSSSNode, SNode, SNode, SNode, Scalar64, |(a, b)| a / b);
binary_op!(PowSSSNode, SNode, SNode, SNode, Scalar64, |(a, b)| {
    f64::powf(a, b)
});
binary_op!(AddSCCNode, SNode, CNode, CNode, CScalar64, |(a, b)| a + b);
binary_op!(SubSCCNode, SNode, CNode, CNode, CScalar64, |(a, b)| a - b);
binary_op!(MulSCCNode, SNode, CNode, CNode, CScalar64, |(a, b)| a * b);
binary_op!(DivSCCNode, SNode, CNode, CNode, CScalar64, |(a, b)| a / b);
binary_op!(PowSCCNode, SNode, CNode, CNode, CScalar64, |(a, b)| {
    f64::powc(a, b)
});

binary_op!(AddCSCNode, CNode, SNode, CNode, CScalar64, |(a, b)| a + b);
binary_op!(SubCSCNode, CNode, SNode, CNode, CScalar64, |(a, b)| a - b);
binary_op!(MulCSCNode, CNode, SNode, CNode, CScalar64, |(a, b)| a * b);
binary_op!(DivCSCNode, CNode, SNode, CNode, CScalar64, |(a, b)| a / b);
binary_op!(PowCSCNode, CNode, SNode, CNode, CScalar64, |(a, b)| {
    CScalar64::powf(a, b)
});

unary_op!(AbsCSNode, CNode, SNode, Scalar64, |a| CScalar64::abs(a));
unary_op!(NormSqrCSNode, CNode, SNode, Scalar64, |a| {
    CScalar64::norm_sqr(&a)
});
unary_op!(RealCSNode, CNode, SNode, Scalar64, |a| CScalar64::re(a));
unary_op!(ImagCSNode, CNode, SNode, Scalar64, |a| CScalar64::im(a));
unary_op!(NegCCNode, CNode, CNode, CScalar64, |a| -a);
binary_op!(AddCCCNode, CNode, CNode, CNode, CScalar64, |(a, b)| a + b);
binary_op!(SubCCCNode, CNode, CNode, CNode, CScalar64, |(a, b)| a - b);
binary_op!(MulCCCNode, CNode, CNode, CNode, CScalar64, |(a, b)| a * b);
binary_op!(DivCCCNode, CNode, CNode, CNode, CScalar64, |(a, b)| a / b);
binary_op!(PowCCCNode, CNode, CNode, CNode, CScalar64, |(a, b)| {
    CScalar64::powc(a, b)
});

// #[derive(Clone)]
// pub struct AddSSNode<T, U>
// where
//     T: SNode,
//     U: SNode,
// {
//     a: T,
//     b: U,
// }
//
// impl<T, U> Dependent for AddSSNode<T, U>
// where
//     T: SNode,
//     U: SNode,
// {
//     fn dependencies(&self) -> Vec<&dyn Resolvable> {
//         vec![&self.a, &self.b]
//     }
// }
// impl<T, U> Resolvable for AddSSNode<T, U>
// where
//     T: SNode,
//     U: SNode,
// {
// }
// impl<T, U> SNode for AddSSNode<T, U>
// where
//     T: SNode,
//     U: SNode,
// {
//     fn eval(&self, ds: &Dataset, pars: &HashMap<String, f64>) -> Vec<Scalar64> {
//         let eval_a = self.a.eval(ds, pars);
//         let eval_b = self.b.eval(ds, pars);
//
//         (eval_a, eval_b)
//             .into_par_iter()
//             .map(|(val_a, val_b)| val_a + val_b)
//             .collect()
//     }
// }
