/// Demonstrates building a GA with custom operators via the builder API.
use evo_engine::algorithms::ga::GeneticAlgorithmBuilder;
use evo_engine::operators::{BlxAlphaCrossover, GaussianMutation, TournamentSelection};
use evo_engine::problems::single_objective::Rosenbrock;
use evo_engine::{EvolutionConfig, EvolutionaryAlgorithm};

fn main() {
    let ga = GeneticAlgorithmBuilder::new()
        .crossover(BlxAlphaCrossover { alpha: 0.5 })
        .mutation(GaussianMutation {
            sigma: 0.3,
            prob: 0.15,
        })
        .selection(TournamentSelection { k: 5 })
        .crossover_prob(0.85)
        .build();

    let config = EvolutionConfig {
        population_size: 150,
        max_generations: 500,
        target_fitness: Some(1e-4),
        elitism_count: 2,
        seed: Some(7),
    };

    let result = ga
        .run(&Rosenbrock { dim: 10 }, &config)
        .expect("optimization failed");

    println!("Best fitness: {:.6e}", result.best.fitness());
    println!("Generations run: {}", result.generations_run);
    println!(
        "First 5 genes: {:?}",
        &result.best.genome[..5]
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );
}
