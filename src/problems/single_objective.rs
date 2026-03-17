//! Classic single-objective benchmark functions.

use crate::{Individual, Problem};
use rand::Rng;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Helper: real-valued problem skeleton
// ---------------------------------------------------------------------------

fn random_real_genome(bounds: &[(f64, f64)], rng: &mut impl Rng) -> Vec<f64> {
    bounds.iter().map(|(lo, hi)| rng.gen::<f64>() * (hi - lo) + lo).collect()
}

// ---------------------------------------------------------------------------
// Rastrigin  f(x) = 10n + Σ[x_i² - 10·cos(2πx_i)]   global min: 0 at origin
// ---------------------------------------------------------------------------

pub struct Rastrigin {
    pub dim: usize,
}

impl Problem for Rastrigin {
    type Genome = Vec<f64>;
    fn num_objectives(&self) -> usize { 1 }
    fn dimension(&self) -> usize { self.dim }
    fn bounds(&self) -> Vec<(f64, f64)> { vec![(-5.12, 5.12); self.dim] }
    fn random_genome(&self, rng: &mut impl Rng) -> Vec<f64> { random_real_genome(&self.bounds(), rng) }

    fn evaluate(&self, ind: &mut Individual<Vec<f64>>) {
        let val: f64 = 10.0 * self.dim as f64
            + ind.genome.iter().map(|x| x * x - 10.0 * (2.0 * PI * x).cos()).sum::<f64>();
        ind.objectives = vec![val];
    }
}

// ---------------------------------------------------------------------------
// Rosenbrock  f(x) = Σ[100(x_{i+1}-x_i²)² + (1-x_i)²]  global min: 0 at (1,..,1)
// ---------------------------------------------------------------------------

pub struct Rosenbrock {
    pub dim: usize,
}

impl Problem for Rosenbrock {
    type Genome = Vec<f64>;
    fn num_objectives(&self) -> usize { 1 }
    fn dimension(&self) -> usize { self.dim }
    fn bounds(&self) -> Vec<(f64, f64)> { vec![(-5.0, 10.0); self.dim] }
    fn random_genome(&self, rng: &mut impl Rng) -> Vec<f64> { random_real_genome(&self.bounds(), rng) }

    fn evaluate(&self, ind: &mut Individual<Vec<f64>>) {
        let val: f64 = ind.genome.windows(2).map(|w| {
            100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2)
        }).sum();
        ind.objectives = vec![val];
    }
}

// ---------------------------------------------------------------------------
// Ackley  global min: 0 at origin
// ---------------------------------------------------------------------------

pub struct Ackley {
    pub dim: usize,
}

impl Problem for Ackley {
    type Genome = Vec<f64>;
    fn num_objectives(&self) -> usize { 1 }
    fn dimension(&self) -> usize { self.dim }
    fn bounds(&self) -> Vec<(f64, f64)> { vec![(-32.768, 32.768); self.dim] }
    fn random_genome(&self, rng: &mut impl Rng) -> Vec<f64> { random_real_genome(&self.bounds(), rng) }

    fn evaluate(&self, ind: &mut Individual<Vec<f64>>) {
        let n = self.dim as f64;
        let sum_sq: f64 = ind.genome.iter().map(|x| x * x).sum::<f64>();
        let sum_cos: f64 = ind.genome.iter().map(|x| (2.0 * PI * x).cos()).sum::<f64>();
        let val = -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + 20.0 + std::f64::consts::E;
        ind.objectives = vec![val];
    }
}

// ---------------------------------------------------------------------------
// Schwefel  global min: 0 at (420.9687, …)
// ---------------------------------------------------------------------------

pub struct Schwefel {
    pub dim: usize,
}

impl Problem for Schwefel {
    type Genome = Vec<f64>;
    fn num_objectives(&self) -> usize { 1 }
    fn dimension(&self) -> usize { self.dim }
    fn bounds(&self) -> Vec<(f64, f64)> { vec![(-500.0, 500.0); self.dim] }
    fn random_genome(&self, rng: &mut impl Rng) -> Vec<f64> { random_real_genome(&self.bounds(), rng) }

    fn evaluate(&self, ind: &mut Individual<Vec<f64>>) {
        let n = self.dim as f64;
        let val = 418.9829 * n
            - ind.genome.iter().map(|x| x * x.abs().sqrt().sin()).sum::<f64>();
        ind.objectives = vec![val];
    }
}

// ---------------------------------------------------------------------------
// Griewank  global min: 0 at origin
// ---------------------------------------------------------------------------

pub struct Griewank {
    pub dim: usize,
}

impl Problem for Griewank {
    type Genome = Vec<f64>;
    fn num_objectives(&self) -> usize { 1 }
    fn dimension(&self) -> usize { self.dim }
    fn bounds(&self) -> Vec<(f64, f64)> { vec![(-600.0, 600.0); self.dim] }
    fn random_genome(&self, rng: &mut impl Rng) -> Vec<f64> { random_real_genome(&self.bounds(), rng) }

    fn evaluate(&self, ind: &mut Individual<Vec<f64>>) {
        let sum: f64 = ind.genome.iter().map(|x| x * x / 4000.0).sum();
        let prod: f64 = ind.genome.iter().enumerate()
            .map(|(i, x)| (x / ((i + 1) as f64).sqrt()).cos())
            .product();
        ind.objectives = vec![sum - prod + 1.0];
    }
}

// ---------------------------------------------------------------------------
// Levy  global min: 0 at (1, …, 1)
// ---------------------------------------------------------------------------

pub struct Levy {
    pub dim: usize,
}

impl Problem for Levy {
    type Genome = Vec<f64>;
    fn num_objectives(&self) -> usize { 1 }
    fn dimension(&self) -> usize { self.dim }
    fn bounds(&self) -> Vec<(f64, f64)> { vec![(-10.0, 10.0); self.dim] }
    fn random_genome(&self, rng: &mut impl Rng) -> Vec<f64> { random_real_genome(&self.bounds(), rng) }

    fn evaluate(&self, ind: &mut Individual<Vec<f64>>) {
        let w: Vec<f64> = ind.genome.iter().map(|x| 1.0 + (x - 1.0) / 4.0).collect();
        let n = w.len();
        let term1 = (PI * w[0]).sin().powi(2);
        let term3 = (w[n - 1] - 1.0).powi(2) * (1.0 + (2.0 * PI * w[n - 1]).sin().powi(2));
        let sum: f64 = w[..n - 1]
            .iter()
            .map(|wi| (wi - 1.0).powi(2) * (1.0 + 10.0 * (PI * wi + 1.0).sin().powi(2)))
            .sum();
        ind.objectives = vec![term1 + sum + term3];
    }
}
