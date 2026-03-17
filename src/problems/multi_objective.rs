//! Multi-objective benchmark problems (ZDT test suite).

use crate::{Individual, Problem};
use rand::Rng;
use std::f64::consts::PI;

fn random_real_genome(bounds: &[(f64, f64)], rng: &mut impl Rng) -> Vec<f64> {
    bounds.iter().map(|(lo, hi)| rng.gen::<f64>() * (hi - lo) + lo).collect()
}

// ---------------------------------------------------------------------------
// ZDT1: convex Pareto front
// ---------------------------------------------------------------------------

pub struct Zdt1 {
    pub dim: usize,
}

impl Problem for Zdt1 {
    type Genome = Vec<f64>;
    fn num_objectives(&self) -> usize { 2 }
    fn dimension(&self) -> usize { self.dim }
    fn bounds(&self) -> Vec<(f64, f64)> { vec![(0.0, 1.0); self.dim] }
    fn random_genome(&self, rng: &mut impl Rng) -> Vec<f64> { random_real_genome(&self.bounds(), rng) }

    fn evaluate(&self, ind: &mut Individual<Vec<f64>>) {
        let f1 = ind.genome[0];
        let g: f64 = 1.0 + 9.0 * ind.genome[1..].iter().sum::<f64>() / (self.dim - 1) as f64;
        let f2 = g * (1.0 - (f1 / g).sqrt());
        ind.objectives = vec![f1, f2];
    }
}

// ---------------------------------------------------------------------------
// ZDT2: non-convex Pareto front
// ---------------------------------------------------------------------------

pub struct Zdt2 {
    pub dim: usize,
}

impl Problem for Zdt2 {
    type Genome = Vec<f64>;
    fn num_objectives(&self) -> usize { 2 }
    fn dimension(&self) -> usize { self.dim }
    fn bounds(&self) -> Vec<(f64, f64)> { vec![(0.0, 1.0); self.dim] }
    fn random_genome(&self, rng: &mut impl Rng) -> Vec<f64> { random_real_genome(&self.bounds(), rng) }

    fn evaluate(&self, ind: &mut Individual<Vec<f64>>) {
        let f1 = ind.genome[0];
        let g: f64 = 1.0 + 9.0 * ind.genome[1..].iter().sum::<f64>() / (self.dim - 1) as f64;
        let f2 = g * (1.0 - (f1 / g).powi(2));
        ind.objectives = vec![f1, f2];
    }
}

// ---------------------------------------------------------------------------
// ZDT3: disconnected Pareto front
// ---------------------------------------------------------------------------

pub struct Zdt3 {
    pub dim: usize,
}

impl Problem for Zdt3 {
    type Genome = Vec<f64>;
    fn num_objectives(&self) -> usize { 2 }
    fn dimension(&self) -> usize { self.dim }
    fn bounds(&self) -> Vec<(f64, f64)> { vec![(0.0, 1.0); self.dim] }
    fn random_genome(&self, rng: &mut impl Rng) -> Vec<f64> { random_real_genome(&self.bounds(), rng) }

    fn evaluate(&self, ind: &mut Individual<Vec<f64>>) {
        let f1 = ind.genome[0];
        let g: f64 = 1.0 + 9.0 * ind.genome[1..].iter().sum::<f64>() / (self.dim - 1) as f64;
        let f2 = g * (1.0 - (f1 / g).sqrt() - (f1 / g) * (10.0 * PI * f1).sin());
        ind.objectives = vec![f1, f2];
    }
}
