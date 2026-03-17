//! NSGA-II: Non-dominated Sorting Genetic Algorithm II.
//!
//! The standard multi-objective optimizer. Maintains a population sorted
//! by Pareto dominance rank and uses crowding distance to preserve
//! diversity along the front.

use crate::operators::crossover::sbx_crossover;
use crate::operators::mutation::polynomial_mutation;
use crate::operators::selection::crowded_tournament_select;
use crate::{
    crowding_distance_assignment, non_dominated_sort, EvoResult, EvolutionConfig,
    EvolutionResult, EvolutionaryAlgorithm, GenerationStats, Individual, Problem,
};
use ordered_float::OrderedFloat;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct Nsga2 {
    pub crossover_prob: f64,
    pub sbx_eta: f64,
    pub pm_eta: f64,
    pub mutation_prob: f64,
}

impl Default for Nsga2 {
    fn default() -> Self {
        Self {
            crossover_prob: 0.9,
            sbx_eta: 15.0,
            pm_eta: 20.0,
            mutation_prob: 0.1,
        }
    }
}

impl EvolutionaryAlgorithm for Nsga2 {
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
        let mut_prob = self.mutation_prob / dim as f64;

        // Initial parent population
        let mut parents: Vec<Individual<Vec<f64>>> = (0..np)
            .map(|_| {
                let g = problem.random_genome(&mut rng);
                let mut ind = Individual::new(g);
                problem.evaluate(&mut ind);
                ind
            })
            .collect();

        let mut history = Vec::new();
        let mut generations_run = 0;

        for gen in 0..config.max_generations {
            generations_run = gen + 1;

            // Assign ranks and crowding to parents for selection
            non_dominated_sort(&mut parents);
            let front_indices: Vec<usize> = (0..parents.len()).collect();
            crowding_distance_assignment(&mut parents, &front_indices);

            // Generate offspring via crossover + mutation
            let mut offspring: Vec<Individual<Vec<f64>>> = Vec::with_capacity(np);

            while offspring.len() < np {
                let p1 = crowded_tournament_select(&parents, &mut rng);
                let p2 = crowded_tournament_select(&parents, &mut rng);

                let (mut c1g, mut c2g) = if rng.gen::<f64>() < self.crossover_prob {
                    sbx_crossover(&p1.genome, &p2.genome, &bounds, self.sbx_eta, &mut rng)
                } else {
                    (p1.genome.clone(), p2.genome.clone())
                };

                polynomial_mutation(&mut c1g, &bounds, self.pm_eta, mut_prob, &mut rng);
                polynomial_mutation(&mut c2g, &bounds, self.pm_eta, mut_prob, &mut rng);

                let mut c1 = Individual::new(c1g);
                problem.evaluate(&mut c1);
                offspring.push(c1);

                if offspring.len() < np {
                    let mut c2 = Individual::new(c2g);
                    problem.evaluate(&mut c2);
                    offspring.push(c2);
                }
            }

            // Combine parents + offspring, then select next generation
            let mut combined: Vec<Individual<Vec<f64>>> = parents;
            combined.extend(offspring);
            non_dominated_sort(&mut combined);

            let max_rank = combined.iter().map(|i| i.rank).max().unwrap_or(0);
            let mut new_parents: Vec<Individual<Vec<f64>>> = Vec::with_capacity(np);

            for rank in 0..=max_rank {
                let front: Vec<usize> = combined
                    .iter()
                    .enumerate()
                    .filter(|(_, ind)| ind.rank == rank)
                    .map(|(i, _)| i)
                    .collect();

                if new_parents.len() + front.len() <= np {
                    crowding_distance_assignment(&mut combined, &front);
                    for &i in &front {
                        new_parents.push(combined[i].clone());
                    }
                } else {
                    // This front doesn't fit entirely; pick by crowding distance
                    crowding_distance_assignment(&mut combined, &front);
                    let mut sorted_front = front;
                    sorted_front.sort_by(|&a, &b| {
                        OrderedFloat(combined[b].crowding_distance)
                            .cmp(&OrderedFloat(combined[a].crowding_distance))
                    });
                    let remaining = np - new_parents.len();
                    for &i in sorted_front.iter().take(remaining) {
                        new_parents.push(combined[i].clone());
                    }
                    break;
                }
            }

            parents = new_parents;

            let best_f = parents
                .iter()
                .map(|i| i.fitness())
                .fold(f64::INFINITY, f64::min);
            let mean_f = parents.iter().map(|i| i.fitness()).sum::<f64>() / np as f64;
            history.push(GenerationStats {
                generation: gen,
                best_fitness: best_f,
                mean_fitness: mean_f,
                worst_fitness: parents
                    .iter()
                    .map(|i| i.fitness())
                    .fold(f64::NEG_INFINITY, f64::max),
                diversity: 0.0,
            });
        }

        // Final sort to extract the Pareto front
        non_dominated_sort(&mut parents);
        let pareto_front: Vec<Individual<Vec<f64>>> =
            parents.iter().filter(|i| i.rank == 0).cloned().collect();

        let best = parents
            .iter()
            .min_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap())
            .unwrap()
            .clone();

        Ok(EvolutionResult {
            best,
            pareto_front,
            history,
            generations_run,
        })
    }
}
