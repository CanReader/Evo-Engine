//! Canonical Genetic Algorithm with configurable operators.

use crate::operators::crossover::sbx_crossover;
use crate::operators::mutation::polynomial_mutation;
use crate::operators::selection::tournament_select;
use crate::{
    population_diversity, EvolutionConfig, EvolutionResult, EvolutionaryAlgorithm,
    GenerationStats, Individual, Problem,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

pub struct GeneticAlgorithm {
    pub crossover_prob: f64,
    pub mutation_prob: f64,
    pub tournament_size: usize,
    pub sbx_eta: f64,
    pub pm_eta: f64,
}

impl Default for GeneticAlgorithm {
    fn default() -> Self {
        Self {
            crossover_prob: 0.9,
            mutation_prob: 0.1,
            tournament_size: 3,
            sbx_eta: 20.0,
            pm_eta: 20.0,
        }
    }
}

impl EvolutionaryAlgorithm for GeneticAlgorithm {
    type Genome = Vec<f64>;

    fn run<P: Problem<Genome = Vec<f64>>>(
        &self,
        problem: &P,
        config: &EvolutionConfig,
    ) -> EvolutionResult<Vec<f64>> {
        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let bounds = problem.bounds();
        let mut_prob_per_gene = self.mutation_prob / problem.dimension() as f64;

        // Initialise population
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

        for gen in 0..config.max_generations {
            // Sort for elitism
            pop.sort_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap());

            // Statistics
            let fitnesses: Vec<f64> = pop.iter().map(|i| i.fitness()).collect();
            let stats = GenerationStats {
                generation: gen,
                best_fitness: fitnesses[0],
                mean_fitness: fitnesses.iter().sum::<f64>() / fitnesses.len() as f64,
                worst_fitness: *fitnesses.last().unwrap(),
                diversity: population_diversity(&pop),
            };
            history.push(stats);

            // Update global best
            if pop[0].fitness() < best_ever.fitness() {
                best_ever = pop[0].clone();
            }

            // Early stop
            if let Some(target) = config.target_fitness {
                if best_ever.fitness() <= target {
                    break;
                }
            }

            // Elitism
            let mut new_pop: Vec<Individual<Vec<f64>>> = pop[..config.elitism_count].to_vec();

            // Breed offspring
            while new_pop.len() < config.population_size {
                let p1 = tournament_select(&pop, self.tournament_size, &mut rng);
                let p2 = tournament_select(&pop, self.tournament_size, &mut rng);

                let (mut c1_genome, mut c2_genome) = if rng.gen::<f64>() < self.crossover_prob {
                    sbx_crossover(&p1.genome, &p2.genome, &bounds, self.sbx_eta, &mut rng)
                } else {
                    (p1.genome.clone(), p2.genome.clone())
                };

                polynomial_mutation(&mut c1_genome, &bounds, self.pm_eta, mut_prob_per_gene, &mut rng);
                polynomial_mutation(&mut c2_genome, &bounds, self.pm_eta, mut_prob_per_gene, &mut rng);

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

        EvolutionResult {
            best: best_ever,
            pareto_front: Vec::new(),
            history,
            generations_run: config.max_generations,
        }
    }
}
