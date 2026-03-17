use rand::{Rng, RngCore};
use rand_distr::{Cauchy, Distribution, Normal};

// ---------------------------------------------------------------------------
// Polynomial mutation (pairs well with SBX crossover)
// ---------------------------------------------------------------------------

/// Bounded polynomial mutation. `eta_m` controls how close the mutant
/// stays to the parent (higher = closer). `prob` is the per-gene
/// mutation probability.
pub fn polynomial_mutation(
    genome: &mut [f64],
    bounds: &[(f64, f64)],
    eta_m: f64,
    prob: f64,
    rng: &mut dyn RngCore,
) {
    for i in 0..genome.len() {
        if rng.gen::<f64>() >= prob {
            continue;
        }
        let (lo, hi) = bounds[i];
        let delta = hi - lo;
        if delta < 1e-30 {
            continue;
        }

        let u: f64 = rng.gen();
        let delta_q = if u < 0.5 {
            let bl = (genome[i] - lo) / delta;
            let val = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - bl).powf(eta_m + 1.0);
            val.powf(1.0 / (eta_m + 1.0)) - 1.0
        } else {
            let bu = (hi - genome[i]) / delta;
            let val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - bu).powf(eta_m + 1.0);
            1.0 - val.powf(1.0 / (eta_m + 1.0))
        };

        genome[i] = (genome[i] + delta_q * delta).clamp(lo, hi);
    }
}

// ---------------------------------------------------------------------------
// Gaussian mutation
// ---------------------------------------------------------------------------

/// Adds a normally distributed perturbation to each gene.
pub fn gaussian_mutation(
    genome: &mut [f64],
    bounds: &[(f64, f64)],
    sigma: f64,
    prob: f64,
    rng: &mut dyn RngCore,
) {
    let normal = Normal::new(0.0, sigma).unwrap();
    for i in 0..genome.len() {
        if rng.gen::<f64>() < prob {
            genome[i] = (genome[i] + normal.sample(rng)).clamp(bounds[i].0, bounds[i].1);
        }
    }
}

// ---------------------------------------------------------------------------
// Cauchy mutation (heavier tails, better at jumping out of local optima)
// ---------------------------------------------------------------------------

/// Similar to Gaussian mutation but uses a Cauchy distribution, which has
/// heavier tails. This means it occasionally produces large jumps that
/// help escape local optima.
pub fn cauchy_mutation(
    genome: &mut [f64],
    bounds: &[(f64, f64)],
    scale: f64,
    prob: f64,
    rng: &mut dyn RngCore,
) {
    let cauchy = Cauchy::new(0.0, scale).unwrap();
    for i in 0..genome.len() {
        if rng.gen::<f64>() < prob {
            genome[i] = (genome[i] + cauchy.sample(rng)).clamp(bounds[i].0, bounds[i].1);
        }
    }
}

// ---------------------------------------------------------------------------
// Adaptive mutation (step size self-adapts via 1/5 success rule)
// ---------------------------------------------------------------------------

/// Tracks mutation step size and adjusts it based on how often mutations
/// actually improve fitness. Implements Rechenberg's 1/5 success rule:
/// if more than 1/5 of mutations improve, increase step size; otherwise shrink it.
pub struct AdaptiveMutator {
    pub sigma: f64,
    pub min_sigma: f64,
    pub max_sigma: f64,
    pub success_count: usize,
    pub total_count: usize,
    pub adapt_interval: usize,
}

impl AdaptiveMutator {
    pub fn new(initial_sigma: f64) -> Self {
        Self {
            sigma: initial_sigma,
            min_sigma: 1e-8,
            max_sigma: 5.0,
            success_count: 0,
            total_count: 0,
            adapt_interval: 20,
        }
    }

    pub fn mutate(&mut self, genome: &mut [f64], bounds: &[(f64, f64)], rng: &mut dyn RngCore) {
        let normal = Normal::new(0.0, self.sigma).unwrap();
        for i in 0..genome.len() {
            genome[i] = (genome[i] + normal.sample(rng)).clamp(bounds[i].0, bounds[i].1);
        }
        self.total_count += 1;
    }

    pub fn report_success(&mut self, improved: bool) {
        if improved {
            self.success_count += 1;
        }
        if self.total_count >= self.adapt_interval && self.total_count > 0 {
            let ratio = self.success_count as f64 / self.total_count as f64;
            if ratio > 0.2 {
                self.sigma = (self.sigma * 1.2).min(self.max_sigma);
            } else {
                self.sigma = (self.sigma * 0.82).max(self.min_sigma);
            }
            self.success_count = 0;
            self.total_count = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Non-uniform mutation (perturbation decreases over generations)
// ---------------------------------------------------------------------------

/// Mutation strength decreases as the run progresses, controlled by
/// the shape parameter `b` (typically 2-5). This lets the algorithm
/// explore broadly at first and fine-tune later.
pub fn non_uniform_mutation(
    genome: &mut [f64],
    bounds: &[(f64, f64)],
    generation: usize,
    max_generation: usize,
    b: f64,
    prob: f64,
    rng: &mut dyn RngCore,
) {
    let t_ratio = generation as f64 / max_generation as f64;
    for i in 0..genome.len() {
        if rng.gen::<f64>() >= prob {
            continue;
        }
        let (lo, hi) = bounds[i];
        let tau: f64 = rng.gen();
        let delta = if rng.gen::<bool>() {
            (hi - genome[i]) * (1.0 - tau.powf((1.0 - t_ratio).powf(b)))
        } else {
            -(genome[i] - lo) * (1.0 - tau.powf((1.0 - t_ratio).powf(b)))
        };
        genome[i] = (genome[i] + delta).clamp(lo, hi);
    }
}
