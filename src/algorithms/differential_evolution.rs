//! Differential Evolution with three strategy variants.
//!
//! DE works differently from GA: instead of selection + crossover + mutation
//! as separate steps, each individual generates a trial vector by combining
//! differences between other population members, then the trial replaces
//! the parent only if it is better (greedy selection).

use crate::{
    population_diversity, EvoResult, EvolutionConfig, EvolutionResult, EvolutionaryAlgorithm,
    GenerationStats, Individual, Problem,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Copy, Debug)]
pub enum DEStrategy {
    /// DE/rand/1/bin: base vector chosen at random.
    Rand1Bin,
    /// DE/best/1/bin: base vector is the current best.
    Best1Bin,
    /// DE/current-to-best/1: base is the current individual,
    /// biased toward the best. Good balance of exploration and exploitation.
    CurrentToBest1,
}

pub struct DifferentialEvolution {
    pub strategy: DEStrategy,
    /// Scale factor controlling the amplification of difference vectors.
    pub f: f64,
    /// Crossover rate for binomial crossover.
    pub cr: f64,
    /// When true, F is randomly perturbed each generation (helps avoid stagnation).
    pub dither: bool,
}

impl Default for DifferentialEvolution {
    fn default() -> Self {
        Self {
            strategy: DEStrategy::Rand1Bin,
            f: 0.8,
            cr: 0.9,
            dither: true,
        }
    }
}

impl EvolutionaryAlgorithm for DifferentialEvolution {
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

        let bounds = problem.bounds();
        let dim = problem.dimension();
        let np = config.population_size;

        let mut pop: Vec<Individual<Vec<f64>>> = (0..np)
            .map(|_| {
                let g = problem.random_genome(&mut rng);
                let mut ind = Individual::new(g);
                problem.evaluate(&mut ind);
                ind
            })
            .collect();

        let mut best_ever = pop
            .iter()
            .min_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap())
            .unwrap()
            .clone();

        let mut history = Vec::new();
        let mut generations_run = 0;

        for gen in 0..config.max_generations {
            generations_run = gen + 1;

            let f = if self.dither {
                self.f + rng.gen::<f64>() * 0.2 - 0.1
            } else {
                self.f
            };

            let best_idx = pop
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.fitness().partial_cmp(&b.fitness()).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            let mut trials: Vec<Individual<Vec<f64>>> = Vec::with_capacity(np);

            for i in 0..np {
                let mut pick = || loop {
                    let idx = rng.gen_range(0..np);
                    if idx != i {
                        return idx;
                    }
                };

                let mutant: Vec<f64> = match self.strategy {
                    DEStrategy::Rand1Bin => {
                        let (a, b, c) = (pick(), pick(), pick());
                        (0..dim)
                            .map(|d| pop[a].genome[d] + f * (pop[b].genome[d] - pop[c].genome[d]))
                            .collect()
                    }
                    DEStrategy::Best1Bin => {
                        let (a, b) = (pick(), pick());
                        (0..dim)
                            .map(|d| {
                                pop[best_idx].genome[d]
                                    + f * (pop[a].genome[d] - pop[b].genome[d])
                            })
                            .collect()
                    }
                    DEStrategy::CurrentToBest1 => {
                        let (a, b) = (pick(), pick());
                        (0..dim)
                            .map(|d| {
                                pop[i].genome[d]
                                    + f * (pop[best_idx].genome[d] - pop[i].genome[d])
                                    + f * (pop[a].genome[d] - pop[b].genome[d])
                            })
                            .collect()
                    }
                };

                // Binomial crossover
                let j_rand = rng.gen_range(0..dim);
                let trial_genome: Vec<f64> = (0..dim)
                    .map(|d| {
                        let val = if rng.gen::<f64>() < self.cr || d == j_rand {
                            mutant[d]
                        } else {
                            pop[i].genome[d]
                        };
                        val.clamp(bounds[d].0, bounds[d].1)
                    })
                    .collect();

                let mut trial = Individual::new(trial_genome);
                problem.evaluate(&mut trial);

                // Greedy selection: keep the better one
                if trial.constrained_cmp(&pop[i]) != std::cmp::Ordering::Greater {
                    trials.push(trial);
                } else {
                    trials.push(pop[i].clone());
                }
            }

            pop = trials;

            for ind in &pop {
                if ind.fitness() < best_ever.fitness() {
                    best_ever = ind.clone();
                }
            }

            let fitnesses: Vec<f64> = pop.iter().map(|i| i.fitness()).collect();
            history.push(GenerationStats {
                generation: gen,
                best_fitness: best_ever.fitness(),
                mean_fitness: fitnesses.iter().sum::<f64>() / np as f64,
                worst_fitness: fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                diversity: population_diversity(&pop),
            });

            if let Some(target) = config.target_fitness {
                if best_ever.fitness() <= target {
                    break;
                }
            }
        }

        Ok(EvolutionResult {
            best: best_ever,
            pareto_front: Vec::new(),
            history,
            generations_run,
        })
    }
}
