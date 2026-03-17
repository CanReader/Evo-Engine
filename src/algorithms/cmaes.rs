//! CMA-ES: Covariance Matrix Adaptation Evolution Strategy.
//!
//! Simplified diagonal-only implementation following Hansen & Ostermeier.
//! The diagonal approximation keeps memory and CPU usage linear in the
//! number of dimensions, which makes it practical for high-D problems
//! at the cost of not modeling variable correlations.

use crate::{
    EvoError, EvoResult, EvolutionConfig, EvolutionResult, EvolutionaryAlgorithm,
    GenerationStats, Individual, Problem,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

pub struct CmaEs {
    /// Initial step size relative to the search space span.
    pub initial_sigma: f64,
}

impl Default for CmaEs {
    fn default() -> Self {
        Self { initial_sigma: 0.5 }
    }
}

impl EvolutionaryAlgorithm for CmaEs {
    type Genome = Vec<f64>;

    fn run<P: Problem<Genome = Vec<f64>>>(
        &self,
        problem: &P,
        config: &EvolutionConfig,
    ) -> EvoResult<EvolutionResult<Vec<f64>>> {
        config.validate()?;

        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let normal = Normal::new(0.0, 1.0).unwrap();

        let n = problem.dimension();
        if n == 0 {
            return Err(EvoError::ZeroDimension);
        }

        let bounds = problem.bounds();
        let lambda = config.population_size;
        let mu = lambda / 2;

        // Recombination weights (log-linear, normalized)
        let raw_weights: Vec<f64> = (0..mu)
            .map(|i| ((mu as f64 + 0.5).ln() - ((i + 1) as f64).ln()).max(0.0))
            .collect();
        let w_sum: f64 = raw_weights.iter().sum();
        let weights: Vec<f64> = raw_weights.iter().map(|w| w / w_sum).collect();
        let mu_eff: f64 = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        // Step-size control parameters
        let c_sigma = (mu_eff + 2.0) / (n as f64 + mu_eff + 5.0);
        let d_sigma = 1.0
            + 2.0 * (((mu_eff - 1.0) / (n as f64 + 1.0)).sqrt() - 1.0).max(0.0)
            + c_sigma;
        let chi_n = (n as f64).sqrt()
            * (1.0 - 1.0 / (4.0 * n as f64) + 1.0 / (21.0 * (n as f64).powi(2)));

        // Covariance adaptation parameters
        let c_c = (4.0 + mu_eff / n as f64) / (n as f64 + 4.0 + 2.0 * mu_eff / n as f64);
        let c1 = 2.0 / ((n as f64 + 1.3).powi(2) + mu_eff);
        let c_mu = (2.0 * (mu_eff - 2.0 + 1.0 / mu_eff)
            / ((n as f64 + 2.0).powi(2) + mu_eff))
            .min(1.0 - c1);

        // Initial state: start from center of search space
        let mid: Vec<f64> = bounds.iter().map(|(lo, hi)| (lo + hi) / 2.0).collect();
        let span: Vec<f64> = bounds.iter().map(|(lo, hi)| hi - lo).collect();
        let mut mean: Vec<f64> = mid.clone();
        let mut sigma = self.initial_sigma * span.iter().cloned().fold(0.0f64, f64::max);

        // Diagonal covariance, evolution paths
        let mut diag_c: Vec<f64> = vec![1.0; n];
        let mut p_sigma: Vec<f64> = vec![0.0; n];
        let mut p_c: Vec<f64> = vec![0.0; n];

        let mut best_ever: Option<Individual<Vec<f64>>> = None;
        let mut history = Vec::new();
        let mut generations_run = 0;

        for gen in 0..config.max_generations {
            generations_run = gen + 1;

            // Sample lambda offspring
            let mut pop: Vec<(Individual<Vec<f64>>, Vec<f64>)> = (0..lambda)
                .map(|_| {
                    let z: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
                    let genome: Vec<f64> = (0..n)
                        .map(|i| {
                            let val = mean[i] + sigma * diag_c[i].sqrt() * z[i];
                            val.clamp(bounds[i].0, bounds[i].1)
                        })
                        .collect();
                    let mut ind = Individual::new(genome);
                    problem.evaluate(&mut ind);
                    (ind, z)
                })
                .collect();

            pop.sort_by(|(a, _), (b, _)| a.fitness().partial_cmp(&b.fitness()).unwrap());

            let gen_best = &pop[0].0;
            if best_ever.is_none() || gen_best.fitness() < best_ever.as_ref().unwrap().fitness() {
                best_ever = Some(gen_best.clone());
            }

            let fitnesses: Vec<f64> = pop.iter().map(|(ind, _)| ind.fitness()).collect();
            history.push(GenerationStats {
                generation: gen,
                best_fitness: fitnesses[0],
                mean_fitness: fitnesses.iter().sum::<f64>() / lambda as f64,
                worst_fitness: *fitnesses.last().unwrap(),
                diversity: sigma,
            });

            if let Some(target) = config.target_fitness {
                if best_ever.as_ref().unwrap().fitness() <= target {
                    break;
                }
            }

            // Weighted mean recombination
            let old_mean = mean.clone();
            for i in 0..n {
                mean[i] = (0..mu).map(|j| weights[j] * pop[j].0.genome[i]).sum();
            }

            // Evolution path update
            let mean_shift: Vec<f64> = (0..n)
                .map(|i| (mean[i] - old_mean[i]) / sigma)
                .collect();

            for i in 0..n {
                p_sigma[i] = (1.0 - c_sigma) * p_sigma[i]
                    + (c_sigma * (2.0 - c_sigma) * mu_eff).sqrt() * mean_shift[i]
                        / diag_c[i].sqrt();
            }

            let p_sigma_norm: f64 = p_sigma.iter().map(|x| x * x).sum::<f64>().sqrt();
            let h_sigma = if p_sigma_norm
                / (1.0 - (1.0 - c_sigma).powi(2 * (gen as i32 + 1))).sqrt()
                < (1.4 + 2.0 / (n as f64 + 1.0)) * chi_n
            {
                1.0
            } else {
                0.0
            };

            for i in 0..n {
                p_c[i] = (1.0 - c_c) * p_c[i]
                    + h_sigma * (c_c * (2.0 - c_c) * mu_eff).sqrt() * mean_shift[i];
            }

            // Diagonal covariance update
            for i in 0..n {
                let rank_one = p_c[i] * p_c[i];
                let rank_mu: f64 = (0..mu)
                    .map(|j| weights[j] * (pop[j].1[i] * pop[j].1[i]))
                    .sum();
                diag_c[i] =
                    (1.0 - c1 - c_mu) * diag_c[i] + c1 * rank_one + c_mu * rank_mu;
                diag_c[i] = diag_c[i].max(1e-20);
            }

            // Step size adaptation
            sigma *= ((c_sigma / d_sigma) * (p_sigma_norm / chi_n - 1.0)).exp();
            sigma = sigma.clamp(1e-20, 1e10);
        }

        Ok(EvolutionResult {
            best: best_ever.unwrap(),
            pareto_front: Vec::new(),
            history,
            generations_run,
        })
    }
}
