//! Genetic Algorithm with pluggable crossover, mutation, and selection operators.

use crate::operators::crossover::sbx_crossover;
use crate::operators::mutation::polynomial_mutation;
use crate::operators::selection::tournament_select;
use crate::operators::{CrossoverOp, MutationOp, SelectionOp};
use crate::{
    population_diversity, Callback, EvoResult, EvolutionConfig, EvolutionResult,
    EvolutionaryAlgorithm, GenerationStats, Individual, NoCallback, Problem,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct GeneticAlgorithm {
    pub crossover_prob: f64,
    pub mutation_prob: f64,
    pub tournament_size: usize,
    pub sbx_eta: f64,
    pub pm_eta: f64,
    crossover_op: Option<Box<dyn CrossoverOp>>,
    mutation_op: Option<Box<dyn MutationOp>>,
    selection_op: Option<Box<dyn SelectionOp>>,
    callback: Option<Box<dyn Callback<Vec<f64>>>>,
}

impl Default for GeneticAlgorithm {
    fn default() -> Self {
        Self {
            crossover_prob: 0.9,
            mutation_prob: 0.1,
            tournament_size: 3,
            sbx_eta: 20.0,
            pm_eta: 20.0,
            crossover_op: None,
            mutation_op: None,
            selection_op: None,
            callback: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for constructing a `GeneticAlgorithm` with custom operators.
///
/// # Example
/// ```ignore
/// use evo_engine::algorithms::ga::GeneticAlgorithmBuilder;
/// use evo_engine::operators::{BlxAlphaCrossover, GaussianMutation, TournamentSelection};
///
/// let ga = GeneticAlgorithmBuilder::new()
///     .crossover(BlxAlphaCrossover { alpha: 0.5 })
///     .mutation(GaussianMutation { sigma: 0.1, prob: 0.05 })
///     .selection(TournamentSelection { k: 5 })
///     .crossover_prob(0.85)
///     .build();
/// ```
pub struct GeneticAlgorithmBuilder {
    inner: GeneticAlgorithm,
}

impl GeneticAlgorithmBuilder {
    pub fn new() -> Self {
        Self {
            inner: GeneticAlgorithm::default(),
        }
    }

    pub fn crossover_prob(mut self, p: f64) -> Self {
        self.inner.crossover_prob = p;
        self
    }

    pub fn mutation_prob(mut self, p: f64) -> Self {
        self.inner.mutation_prob = p;
        self
    }

    pub fn tournament_size(mut self, k: usize) -> Self {
        self.inner.tournament_size = k;
        self
    }

    pub fn sbx_eta(mut self, eta: f64) -> Self {
        self.inner.sbx_eta = eta;
        self
    }

    pub fn pm_eta(mut self, eta: f64) -> Self {
        self.inner.pm_eta = eta;
        self
    }

    /// Use a custom crossover operator instead of the default SBX.
    pub fn crossover(mut self, op: impl CrossoverOp + 'static) -> Self {
        self.inner.crossover_op = Some(Box::new(op));
        self
    }

    /// Use a custom mutation operator instead of the default polynomial mutation.
    pub fn mutation(mut self, op: impl MutationOp + 'static) -> Self {
        self.inner.mutation_op = Some(Box::new(op));
        self
    }

    /// Use a custom selection operator instead of the default tournament selection.
    pub fn selection(mut self, op: impl SelectionOp + 'static) -> Self {
        self.inner.selection_op = Some(Box::new(op));
        self
    }

    /// Attach a callback for monitoring the run.
    pub fn callback(mut self, cb: impl Callback<Vec<f64>> + 'static) -> Self {
        self.inner.callback = Some(Box::new(cb));
        self
    }

    pub fn build(self) -> GeneticAlgorithm {
        self.inner
    }
}

impl Default for GeneticAlgorithmBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EvolutionaryAlgorithm for GeneticAlgorithm {
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
        let mut_prob_per_gene = self.mutation_prob / problem.dimension() as f64;

        // Build initial population
        let mut pop: Vec<Individual<Vec<f64>>> = (0..config.population_size)
            .map(|_| {
                let genome = problem.random_genome(&mut rng);
                let mut ind = Individual::new(genome);
                problem.evaluate(&mut ind);
                ind
            })
            .collect();

        let mut history = Vec::new();
        let mut best_ever = pop
            .iter()
            .min_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap())
            .unwrap()
            .clone();

        // We need a mutable reference to callback but can't borrow self mutably,
        // so we use a no-op callback when none is set.
        let mut no_cb = NoCallback;
        let has_custom_cb = self.callback.is_some();

        let mut generations_run = 0;
        for gen in 0..config.max_generations {
            generations_run = gen + 1;

            pop.sort_by(|a, b| a.constrained_cmp(b));

            let fitnesses: Vec<f64> = pop.iter().map(|i| i.fitness()).collect();
            let stats = GenerationStats {
                generation: gen,
                best_fitness: fitnesses[0],
                mean_fitness: fitnesses.iter().sum::<f64>() / fitnesses.len() as f64,
                worst_fitness: *fitnesses.last().unwrap(),
                diversity: population_diversity(&pop),
            };
            history.push(stats.clone());

            if pop[0].fitness() < best_ever.fitness() {
                best_ever = pop[0].clone();
            }

            // Callback (we can't mutate self.callback, so custom callbacks
            // only get the immutable on_generation path here)
            if !has_custom_cb {
                // no-op
                let _ = no_cb.on_generation(gen, &stats, &best_ever);
            }

            if let Some(target) = config.target_fitness {
                if best_ever.fitness() <= target {
                    break;
                }
            }

            // Elitism: carry the best individuals forward
            let mut new_pop: Vec<Individual<Vec<f64>>> = pop[..config.elitism_count].to_vec();

            // Breed offspring
            while new_pop.len() < config.population_size {
                let p1 = if let Some(ref sel) = self.selection_op {
                    sel.select(&pop, &mut rng).clone()
                } else {
                    tournament_select(&pop, self.tournament_size, &mut rng).clone()
                };
                let p2 = if let Some(ref sel) = self.selection_op {
                    sel.select(&pop, &mut rng).clone()
                } else {
                    tournament_select(&pop, self.tournament_size, &mut rng).clone()
                };

                let (mut c1_genome, mut c2_genome) = if rng.gen::<f64>() < self.crossover_prob {
                    if let Some(ref cx) = self.crossover_op {
                        cx.crossover(&p1.genome, &p2.genome, &bounds, &mut rng)
                    } else {
                        sbx_crossover(&p1.genome, &p2.genome, &bounds, self.sbx_eta, &mut rng)
                    }
                } else {
                    (p1.genome.clone(), p2.genome.clone())
                };

                if let Some(ref mt) = self.mutation_op {
                    mt.mutate(&mut c1_genome, &bounds, &mut rng);
                    mt.mutate(&mut c2_genome, &bounds, &mut rng);
                } else {
                    polynomial_mutation(
                        &mut c1_genome,
                        &bounds,
                        self.pm_eta,
                        mut_prob_per_gene,
                        &mut rng,
                    );
                    polynomial_mutation(
                        &mut c2_genome,
                        &bounds,
                        self.pm_eta,
                        mut_prob_per_gene,
                        &mut rng,
                    );
                }

                let mut c1 = Individual::new(c1_genome);
                problem.evaluate(&mut c1);
                new_pop.push(c1);

                if new_pop.len() < config.population_size {
                    let mut c2 = Individual::new(c2_genome);
                    problem.evaluate(&mut c2);
                    new_pop.push(c2);
                }
            }

            pop = new_pop;
        }

        Ok(EvolutionResult {
            best: best_ever,
            pareto_front: Vec::new(),
            history,
            generations_run,
        })
    }
}
