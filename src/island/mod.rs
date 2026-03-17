//! Island Model: parallel sub-populations with periodic migration.
//!
//! Each island evolves independently. Every `migration_interval` generations,
//! the best individuals from each island migrate to the next one in a ring.
//! This keeps diversity high while still allowing good solutions to spread.

use crate::operators::crossover::sbx_crossover;
use crate::operators::mutation::polynomial_mutation;
use crate::operators::selection::tournament_select;
use crate::{
    population_diversity, EvoResult, EvolutionConfig, EvolutionResult, GenerationStats,
    Individual, Problem,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct IslandModelConfig {
    pub num_islands: usize,
    pub migration_interval: usize,
    pub migration_count: usize,
    pub base_config: EvolutionConfig,
    pub crossover_prob: f64,
    pub mutation_prob: f64,
    pub sbx_eta: f64,
    pub pm_eta: f64,
    pub tournament_size: usize,
}

impl Default for IslandModelConfig {
    fn default() -> Self {
        Self {
            num_islands: 4,
            migration_interval: 25,
            migration_count: 3,
            base_config: EvolutionConfig::default(),
            crossover_prob: 0.9,
            mutation_prob: 0.1,
            sbx_eta: 20.0,
            pm_eta: 20.0,
            tournament_size: 3,
        }
    }
}

/// Run the island model. Returns the overall best individual found
/// across all islands and all generations.
pub fn run_island_model<P: Problem<Genome = Vec<f64>>>(
    problem: &P,
    config: &IslandModelConfig,
) -> EvoResult<EvolutionResult<Vec<f64>>> {
    config.base_config.validate()?;

    let island_size = config.base_config.population_size / config.num_islands;
    let bounds = problem.bounds();
    let mut rng = match config.base_config.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let mut_prob_gene = config.mutation_prob / problem.dimension() as f64;

    // Create islands
    let mut islands: Vec<Vec<Individual<Vec<f64>>>> = (0..config.num_islands)
        .map(|_| {
            (0..island_size)
                .map(|_| {
                    let g = problem.random_genome(&mut rng);
                    let mut ind = Individual::new(g);
                    problem.evaluate(&mut ind);
                    ind
                })
                .collect()
        })
        .collect();

    let mut best_ever: Individual<Vec<f64>> = islands
        .iter()
        .flatten()
        .min_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap())
        .unwrap()
        .clone();

    let mut history = Vec::new();
    let mut generations_run = 0;

    for gen in 0..config.base_config.max_generations {
        generations_run = gen + 1;

        // Evolve each island for one generation
        for island in &mut islands {
            island.sort_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap());

            let mut new_pop: Vec<Individual<Vec<f64>>> =
                island[..2.min(island.len())].to_vec();

            while new_pop.len() < island_size {
                let p1 = tournament_select(island, config.tournament_size, &mut rng);
                let p2 = tournament_select(island, config.tournament_size, &mut rng);

                let (mut c1g, mut c2g) = if rng.gen::<f64>() < config.crossover_prob {
                    sbx_crossover(&p1.genome, &p2.genome, &bounds, config.sbx_eta, &mut rng)
                } else {
                    (p1.genome.clone(), p2.genome.clone())
                };

                polynomial_mutation(&mut c1g, &bounds, config.pm_eta, mut_prob_gene, &mut rng);
                polynomial_mutation(&mut c2g, &bounds, config.pm_eta, mut_prob_gene, &mut rng);

                let mut c1 = Individual::new(c1g);
                problem.evaluate(&mut c1);
                new_pop.push(c1);

                if new_pop.len() < island_size {
                    let mut c2 = Individual::new(c2g);
                    problem.evaluate(&mut c2);
                    new_pop.push(c2);
                }
            }

            *island = new_pop;
        }

        // Ring migration: best from island i go to island (i+1) % n
        if gen > 0 && gen % config.migration_interval == 0 {
            let num = config.num_islands;
            let migrants: Vec<Vec<Individual<Vec<f64>>>> = islands
                .iter()
                .map(|island| {
                    let mut sorted = island.clone();
                    sorted.sort_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap());
                    sorted.into_iter().take(config.migration_count).collect()
                })
                .collect();

            // Replace worst individuals on the receiving island
            for i in 0..num {
                let target = (i + 1) % num;
                let island = &mut islands[target];
                island.sort_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap());
                let len = island.len();
                for (j, migrant) in migrants[i].iter().enumerate() {
                    if j < len {
                        island[len - 1 - j] = migrant.clone();
                    }
                }
            }
        }

        // Collect global stats
        let all: Vec<&Individual<Vec<f64>>> = islands.iter().flatten().collect();
        let gen_best = all
            .iter()
            .min_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap())
            .unwrap();
        if gen_best.fitness() < best_ever.fitness() {
            best_ever = (*gen_best).clone();
        }

        let fitnesses: Vec<f64> = all.iter().map(|i| i.fitness()).collect();
        let all_owned: Vec<Individual<Vec<f64>>> = all.into_iter().cloned().collect();
        history.push(GenerationStats {
            generation: gen,
            best_fitness: best_ever.fitness(),
            mean_fitness: fitnesses.iter().sum::<f64>() / fitnesses.len() as f64,
            worst_fitness: fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            diversity: population_diversity(&all_owned),
        });

        if let Some(target) = config.base_config.target_fitness {
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
