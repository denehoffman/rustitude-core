use argmin::core::CostFunction;

pub trait Function {
    fn call(&self, x: &[f64]) -> f64;
    fn derivative(&self, i: usize, x: &[f64]) -> f64 {
        let mut dx = vec![0.0; x.len()];
        let h = f64::sqrt(f64::EPSILON);
        dx[i] = h;
        let x_plus_dx: Vec<f64> = x.iter().zip(dx.iter()).map(|(x, dx)| x + dx).collect();
        let x_minus_dx: Vec<f64> = x.iter().zip(dx).map(|(x, dx)| x - dx).collect();
        (self.call(&x_plus_dx) - self.call(&x_minus_dx)) / (2.0 * h)
    }
    fn second_derivative(&self, i: usize, x: &[f64]) -> f64 {
        let mut dx = vec![0.0; x.len()];
        let h = f64::sqrt(f64::EPSILON);
        dx[i] = h;
        let x_plus_dx: Vec<f64> = x.iter().zip(dx.iter()).map(|(x, dx)| x + dx).collect();
        let x_minus_dx: Vec<f64> = x.iter().zip(dx).map(|(x, dx)| x - dx).collect();
        (self.call(&x_plus_dx) - 2.0 * self.call(x) + self.call(&x_minus_dx)) / (4.0 * h * h)
    }
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        (0..x.len()).map(|i| self.derivative(i, x)).collect()
    }
    fn second_gradient(&self, x: &[f64]) -> Vec<f64> {
        (0..x.len()).map(|i| self.second_derivative(i, x)).collect()
    }
    fn hessian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        (0..x.len())
            .map(|i| {
                (0..x.len())
                    .map(|j| {
                        let f_i = |p| self.derivative(i, p);
                        let mut dx = vec![0.0; x.len()];
                        let h = f64::sqrt(f64::EPSILON);
                        dx[j] = h;
                        let x_plus_dx: Vec<f64> =
                            x.iter().zip(dx.iter()).map(|(x, dx)| x + dx).collect();
                        let x_minus_dx: Vec<f64> = x.iter().zip(dx).map(|(x, dx)| x - dx).collect();
                        (f_i(&x_plus_dx) - f_i(&x_minus_dx)) / (2.0 * h)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

pub trait Algorithm {
    type State;
    type ErrorType;
    fn step(
        &self,
        state: Self::State,
        function: &impl Function,
    ) -> Result<Self::State, Self::ErrorType>;
    fn fit(
        &self,
        initial_state: Self::State,
        function: &impl Function,
        max_iters: usize,
    ) -> Result<Self::State, Self::ErrorType> {
        let mut state = initial_state;
        for _ in 0..max_iters {
            state = self.step(state, function)?;
        }
        Ok(state)
    }
}

pub struct Bound(f64, f64);
impl Bound {
    fn rand(&self) -> f64 {
        (fastrand::f64() * (self.1 - self.0)) + self.0
    }
}

pub struct BasicState {
    pub x: Vec<f64>,
    pub fx: f64,
}

impl BasicState {
    pub fn new(x: Vec<f64>, f: &impl Function) -> Self {
        Self { fx: f.call(&x), x }
    }
}

pub struct BoundedState {
    pub x: Vec<f64>,
    pub fx: f64,
    pub bounds: Vec<Bound>,
}

impl BoundedState {
    pub fn new(x: Vec<f64>, bounds: Vec<Bound>, f: &impl Function) -> Self {
        Self {
            fx: f.call(&x),
            x,
            bounds,
        }
    }
}

pub struct RandomSearch;
impl Algorithm for RandomSearch {
    type State = BoundedState;
    type ErrorType = ();

    fn step(
        &self,
        state: Self::State,
        function: &impl Function,
    ) -> Result<Self::State, Self::ErrorType> {
        let next_x: Vec<f64> = state.bounds.iter().map(Bound::rand).collect();
        let next_fx = function.call(&next_x);
        if next_fx < state.fx {
            Ok(Self::State {
                x: next_x,
                fx: next_fx,
                bounds: state.bounds,
            })
        } else {
            Ok(state)
        }
    }
}

pub struct NewtonRaphson;
impl Algorithm for NewtonRaphson {
    type State = BasicState;

    type ErrorType = (); // TODO:

    fn step(
        &self,
        state: Self::State,
        function: &impl Function,
    ) -> Result<Self::State, Self::ErrorType> {
        let x: Vec<f64> = state
            .x
            .iter()
            .enumerate()
            .map(|(i, xi)| {
                xi - 1.0 * function.derivative(i, &state.x)
                    / function.second_derivative(i, &state.x)
            })
            .collect();
        Ok(Self::State {
            fx: function.call(&x),
            x,
        })
    }
}
