/// Shows how to export run history to CSV and JSON for analysis or plotting.
use evo_engine::algorithms::differential_evolution::{DEStrategy, DifferentialEvolution};
use evo_engine::problems::single_objective::Rastrigin;
use evo_engine::{EvolutionConfig, EvolutionaryAlgorithm};

fn main() {
    let de = DifferentialEvolution {
        strategy: DEStrategy::CurrentToBest1,
        f: 0.7,
        cr: 0.9,
        dither: true,
    };

    let result = de
        .run(
            &Rastrigin { dim: 10 },
            &EvolutionConfig {
                population_size: 100,
                max_generations: 200,
                target_fitness: None,
                elitism_count: 0,
                seed: Some(42),
            },
        )
        .expect("optimization failed");

    // CSV output (pipe to a file: cargo run --example csv_export > history.csv)
    println!("--- CSV ---");
    print!("{}", result.history_to_csv());

    // JSON output (requires the "serialize" feature, enabled by default)
    #[cfg(feature = "serialize")]
    {
        println!("\n--- JSON (first 3 entries) ---");
        let json = result.history_to_json().expect("json serialization failed");
        // Just print the first few lines so it doesn't flood the terminal
        for line in json.lines().take(20) {
            println!("{}", line);
        }
        println!("...");
    }
}
