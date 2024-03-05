use ndarray::array;
use rustitude::{
    gluex::{self, open_gluex},
    prelude::*,
};
fn main() {
    let f0 = gluex::KMatrixConstants {
        g: array![
            [0.74987, 0.06401, -0.23417, 0.0157, -0.14242],
            [-0.01257, 0.00204, -0.01032, 0.267, 0.2278],
            [0.02736, 0.77413, 0.72283, 0.09214, 0.15981],
            [-0.15102, 0.50999, 0.11934, 0.02742, 0.16272],
            [0.36103, 0.13112, 0.36792, -0.04025, -0.17397]
        ],
        m: array![0.51461, 0.90630, 1.23089, 1.46104, 1.69611],
        c: array![
            [0.03728, 0.00000, -0.01398, -0.02203, 0.01397],
            [0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
            [-0.01398, 0.00000, 0.02349, 0.03101, -0.04003],
            [-0.02203, 0.00000, 0.03101, -0.13769, -0.06722],
            [0.01397, 0.00000, -0.04003, -0.06722, -0.28401],
        ],
        m1: array![0.13498, 0.26995, 0.49368, 0.54786, 0.54786],
        m2: array![0.13498, 0.26995, 0.49761, 0.54786, 0.95778],
        n_channels: 5,
        n_resonances: 5,
        wave: gluex::Wave::S,
    };
    let f0_adler = gluex::AdlerZero {
        s_0: 0.009_112_5,
        s_norm: 1.0,
    };
    let f2 = gluex::KMatrixConstants {
        g: array![
            [0.40033, 0.0182, -0.06709, -0.49924],
            [0.15479, 0.173, 0.22941, 0.19295],
            [-0.089, 0.32393, -0.43133, 0.27975],
            [-0.00113, 0.15256, 0.23721, -0.03987]
        ],
        m: array![1.15299, 1.48359, 1.72923, 1.96700],
        c: array![
            [-0.04319, 0.00000, 0.00984, 0.01028],
            [0.00000, 0.00000, 0.00000, 0.00000],
            [0.00984, 0.00000, -0.07344, 0.05533],
            [0.01028, 0.00000, 0.05533, -0.05183],
        ],
        m1: array![0.13498, 0.26995, 0.49368, 0.54786],
        m2: array![0.13498, 0.26995, 0.49761, 0.54786],
        n_channels: 4,
        n_resonances: 4,
        wave: gluex::Wave::D,
    };

    let a0 = gluex::KMatrixConstants {
        g: array![[0.43215, 0.19], [-0.28825, 0.43372]],
        m: array![0.95395, 1.26767],
        c: array![[0.00000, 0.00000], [0.00000, 0.00000],],
        m1: array![0.13498, 0.49368],
        m2: array![0.54786, 0.49761],
        n_channels: 2,
        n_resonances: 2,
        wave: gluex::Wave::S,
    };

    let a2 = gluex::KMatrixConstants {
        g: array![[0.30073, 0.68567], [0.21426, 0.12543], [-0.09162, 0.00184]],
        m: array![1.30080, 1.75351],
        c: array![
            [-0.40184, 0.00033, -0.08707],
            [0.00033, -0.21416, -0.06193],
            [-0.08707, -0.06193, -0.17435],
        ],
        m1: array![0.13498, 0.49368, 0.13498],
        m2: array![0.54786, 0.49761, 0.95778],
        n_channels: 3,
        n_resonances: 2,
        wave: gluex::Wave::D,
    };

    let mut m = Manager::default();
    let data = open_gluex("acc_pol.parquet", true).unwrap();
    m.register(
        "F0+ Re",
        "Re Zlm 0 0 +",
        gluex::Zlm::new(
            gluex::Wave::S0,
            gluex::Reflectivity::Positive,
            gluex::Part::Real,
        ),
    );
    m.register(
        "F0+ Re",
        "F0+",
        gluex::FrozenKMatrix::new(2, &f0, Some(&f0_adler)),
    );
    m.register(
        "F0+ Im",
        "Im Zlm 0 0 +",
        gluex::Zlm::new(
            gluex::Wave::S0,
            gluex::Reflectivity::Positive,
            gluex::Part::Imag,
        ),
    );
    m.register(
        "F0+ Im",
        "F0+",
        gluex::FrozenKMatrix::new(2, &f0, Some(&f0_adler)),
    );
    m.register(
        "A0+ Re",
        "Re Zlm 0 0 +",
        gluex::Zlm::new(
            gluex::Wave::S0,
            gluex::Reflectivity::Positive,
            gluex::Part::Real,
        ),
    );
    m.register("A0+ Re", "A0+", gluex::FrozenKMatrix::new(1, &a0, None));
    m.register(
        "A0+ Im",
        "Im Zlm 0 0 +",
        gluex::Zlm::new(
            gluex::Wave::S0,
            gluex::Reflectivity::Positive,
            gluex::Part::Imag,
        ),
    );
    m.register("A0+ Im", "A0+", gluex::FrozenKMatrix::new(1, &a0, None));
    m.register(
        "F0- Re",
        "Re Zlm 0 0 -",
        gluex::Zlm::new(
            gluex::Wave::S0,
            gluex::Reflectivity::Negative,
            gluex::Part::Real,
        ),
    );
    m.register(
        "F0- Re",
        "F0-",
        gluex::FrozenKMatrix::new(2, &f0, Some(&f0_adler)),
    );
    m.register(
        "F0- Im",
        "Im Zlm 0 0 -",
        gluex::Zlm::new(
            gluex::Wave::S0,
            gluex::Reflectivity::Negative,
            gluex::Part::Imag,
        ),
    );
    m.register(
        "F0- Im",
        "F0-",
        gluex::FrozenKMatrix::new(2, &f0, Some(&f0_adler)),
    );
    m.register(
        "A0- Re",
        "Re Zlm 0 0 -",
        gluex::Zlm::new(
            gluex::Wave::S0,
            gluex::Reflectivity::Negative,
            gluex::Part::Real,
        ),
    );
    m.register("A0- Re", "A0-", gluex::FrozenKMatrix::new(1, &a0, None));
    m.register(
        "A0- Im",
        "Im Zlm 0 0 -",
        gluex::Zlm::new(
            gluex::Wave::S0,
            gluex::Reflectivity::Negative,
            gluex::Part::Imag,
        ),
    );
    m.register("A0- Im", "A0-", gluex::FrozenKMatrix::new(1, &a0, None));
    m.register(
        "F2+ Re",
        "Re Zlm 2 2 +",
        gluex::Zlm::new(
            gluex::Wave::D2,
            gluex::Reflectivity::Positive,
            gluex::Part::Real,
        ),
    );
    m.register("F2+ Re", "F2+", gluex::FrozenKMatrix::new(2, &f2, None));
    m.register(
        "F2+ Im",
        "Im Zlm 2 2 +",
        gluex::Zlm::new(
            gluex::Wave::D2,
            gluex::Reflectivity::Positive,
            gluex::Part::Imag,
        ),
    );
    m.register("F2+ Im", "F2+", gluex::FrozenKMatrix::new(2, &f2, None));
    m.register(
        "A2+ Re",
        "Re Zlm 2 2 +",
        gluex::Zlm::new(
            gluex::Wave::D2,
            gluex::Reflectivity::Positive,
            gluex::Part::Real,
        ),
    );
    m.register("A2+ Re", "A2+", gluex::FrozenKMatrix::new(1, &a2, None));
    m.register(
        "A2+ Im",
        "Im Zlm 2 2 +",
        gluex::Zlm::new(
            gluex::Wave::D2,
            gluex::Reflectivity::Positive,
            gluex::Part::Imag,
        ),
    );
    m.register("A2+ Im", "A2+", gluex::FrozenKMatrix::new(1, &a2, None));

    m.constrain("F0+ Re", "F0+ Im");
    m.constrain("A0+ Re", "A0+ Im");
    m.constrain("F0- Re", "F0- Im");
    m.constrain("A0- Re", "A0- Im");
    m.constrain("F2+ Re", "F2+ Im");
    m.constrain("A2+ Re", "A2+ Im");
    m.fix_imag("F0+ Re");
    m.fix_imag("F0- Re");
    m.sum(vec!["F0+ Re", "A0+ Re", "F2+ Re", "A2+ Re"]);
    m.sum(vec!["F0+ Im", "A0+ Im", "F2+ Im", "A2+ Im"]);
    m.sum(vec!["F0- Re", "A0- Re"]);
    m.sum(vec!["F0- Im", "A0- Im"]);
    println!("{m}");
    dbg!(m.parameters.free_parameters());
    use std::time::Instant;
    let now = Instant::now();
    let res: f64 = m.evaluate(vec![1.0; 50], &data).iter().sum();
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
    println!("{res}");
    let now = Instant::now();
    let res: f64 = m.evaluate(vec![1.0; 50], &data).iter().sum();
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
    println!("{res}");
}

#[cfg(test)]
mod tests {
    use std::fmt::Display;

    use num_complex::Complex64;

    use rustc_hash::FxHashMap as HashMap;
    use rustitude::prelude::*;

    struct A;
    impl Amplitude for A {
        fn evaluate(
            &mut self,
            _dataset: &Dataset,
            _parameters: &HashMap<String, f64>,
        ) -> Vec<Complex64> {
            vec![Complex64::new(1.0, -2.0)]
        }
    }
    impl Display for A {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "A")
        }
    }

    struct B;
    impl Amplitude for B {
        fn evaluate(
            &mut self,
            _dataset: &Dataset,
            _parameters: &HashMap<String, f64>,
        ) -> Vec<Complex64> {
            vec![Complex64::new(1.0, -2.0)]
        }
    }
    impl Display for B {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "B")
        }
    }

    struct C;
    impl Amplitude for C {
        fn evaluate(
            &mut self,
            _dataset: &Dataset,
            parameters: &HashMap<String, f64>,
        ) -> Vec<Complex64> {
            vec![Complex64::new(parameters["par1"], parameters["par2"])]
        }
        fn parameter_names(&self) -> Option<Vec<String>> {
            Some(vec!["par1".to_string(), "par2".to_string()])
        }
    }
    impl Display for C {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "C")
        }
    }

    #[test]
    fn test_parameter_registration() {
        let mut m = Manager::default();
        m.parameters.register("A");
        m.parameters.register("B");
        m.parameters.register("C");
        assert_eq!(
            m.parameters.parameters,
            vec![
                vec!["A".to_string()],
                vec!["B".to_string()],
                vec!["C".to_string()]
            ]
        );
    }

    #[test]
    fn test_parameter_constraint() {
        let mut m = Manager::default();
        m.parameters.register("A");
        m.parameters.register("B");
        m.parameters.register("C");
        m.parameters.constrain("A", "B");
        assert_eq!(
            m.parameters.parameters,
            vec![
                vec!["A".to_string(), "B".to_string()],
                vec!["C".to_string()]
            ]
        );
    }

    #[test]
    fn test_parameter_fix() {
        let mut m = Manager::default();
        m.parameters.register("A");
        m.parameters.fix("A", 3.2);
        assert_eq!(m.parameters.fixed.get("A"), Some(&3.2));
    }

    #[test]
    fn test_parameter_constrain_then_fix() {
        let mut m = Manager::default();
        m.parameters.register("A");
        m.parameters.register("B");
        m.parameters.register("C");
        m.parameters.constrain("A", "B");
        m.parameters.fix("A", 3.2);
        assert_eq!(m.parameters.fixed.get("A"), Some(&3.2));
        assert_eq!(m.parameters.fixed.get("B"), Some(&3.2));
    }

    #[test]
    fn test_parameter_fix_then_constrain() {
        let mut m = Manager::default();
        m.parameters.register("A");
        m.parameters.register("B");
        m.parameters.register("C");
        m.parameters.fix("A", 3.2);
        m.parameters.constrain("A", "B");
        assert_eq!(m.parameters.fixed.get("A"), Some(&3.2));
        assert_eq!(m.parameters.fixed.get("B"), Some(&3.2));
    }

    #[test]
    fn test_parameter_fix_then_register_and_constrain() {
        let mut m = Manager::default();
        m.parameters.register("A");
        m.parameters.register("C");
        m.parameters.fix("A", 3.2);
        m.parameters.register("B");
        m.parameters.constrain("A", "B");
        assert_eq!(m.parameters.fixed.get("A"), Some(&3.2));
        assert_eq!(m.parameters.fixed.get("B"), Some(&3.2));
    }

    #[test]
    fn test_manager_registration() {
        let mut m = Manager::default();
        let a = A;
        let b = B;
        m.register("a", "a", a);
        m.register("a", "b", b);
    }

    #[test]
    fn test_manager_constrain() {
        let mut m = Manager::default();
        let a = A;
        let b = B;
        m.register("a", "a", a);
        m.register("b", "b", b);
        m.constrain("a", "b");
        assert_eq!(
            m.parameters.parameters,
            vec![
                vec!["a re".to_string(), "b re".to_string()],
                vec!["a im".to_string(), "b im".to_string()],
                vec!["a scale".to_string()],
                vec!["b scale".to_string()]
            ]
        )
    }

    #[test]
    fn test_manager_set_anchor() {
        let mut m = Manager::default();
        let a = A;
        let b = B;
        m.register("a", "a", a);
        m.register("b", "b", b);
        m.fix_imag("a");
        assert_eq!(m.parameters.fixed.get("a im"), Some(&0.0));
    }

    #[test]
    fn test_manager_constrain_and_set_anchor() {
        let mut m = Manager::default();
        let a = A;
        let b = B;
        m.register("a", "a", a);
        m.register("b", "b", b);
        m.constrain("a", "b");
        m.fix_imag("a");
        assert_eq!(m.parameters.fixed.get("a im"), Some(&0.0));
        assert_eq!(m.parameters.fixed.get("b im"), Some(&0.0));
    }

    #[test]
    fn test_manager_set_anchor_and_constrain() {
        let mut m = Manager::default();
        let a = A;
        let b = B;
        m.register("a", "a", a);
        m.register("b", "b", b);
        m.fix_imag("a");
        m.constrain("a", "b");
        assert_eq!(m.parameters.fixed.get("a im"), Some(&0.0));
        assert_eq!(m.parameters.fixed.get("b im"), Some(&0.0));
    }

    #[test]
    fn test_parameterized_amplitude() {
        let mut m = Manager::default();
        let c = C;
        m.register("group_c", "amp_c", c);
        m.fix_imag("group_c");
        m.sum(vec!["group_c"]);
        let dataset = Dataset::from_size(1);
        let res = m.evaluate(vec![5.0, 10.0, 1.0], &dataset);
        assert_eq!(res[0], 125.0); // |5+10i|^2 = 125
    }
}
