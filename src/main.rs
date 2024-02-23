#![allow(unused_imports)]
use rustc_hash::FxHashMap as HashMap;

trait Amplitude {}

struct Manager {
    amplitudes: HashMap<String, Box<dyn Amplitude>>,
    groups: HashMap<String, Vec<String>>,
    sums: HashMap<String, Vec<String>>,
}

impl Manager {}

fn main() {}

//
// |AB + CDE| + |AE + F| + |F|
//
// Amplitudes:
// A, B, C, D, E, F
//
// Groups:
//         AB,  CDE, AE,  F
//         ||   |||  ||   |--
// Sums:   ||   |||  ||   |  |
//         AB + CDE, AE + F, F
//
// Parameters:
// {x} re, {x} im for {x} in Groups, pass amplitude-specific parameter names in constructor,
// then we can just have the user call pars[self.parname] to get it, and the names can be
// registered when the amplitude is constructed
//
// Constraints:
// |AE + F| -> F == |F| -> F <=> Group equality
