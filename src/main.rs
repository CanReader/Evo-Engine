//! evo-engine: Advanced Evolutionary Computation Framework
//!
//! Demonstrates all algorithms on benchmark problems.

use evo_engine::algorithms::cmaes::CmaEs;
use evo_engine::algorithms::differential_evolution::{DEStrategy, DifferentialEvolution};
use evo_engine::algorithms::ga::GeneticAlgorithm;
use evo_engine::algorithms::nsga2::Nsga2;
use evo_engine::island::{run_island_model, IslandModelConfig};
use evo_engine::problems::multi_objective::Zdt1;
use evo_engine::problems::single_objective::*;
use evo_engine::{EvolutionConfig, EvolutionResult, EvolutionaryAlgorithm};

fn report(name: &str, result: &EvolutionResult<Vec<f64>>) {
    println!("┌─────────────────────────────────────────────────────────");
    println!("│ {}", name);
    println!("├─────────────────────────────────────────────────────────");
    println!("│ Best fitness : {:.8e}", result.best.fitness());
    println!(
        "│ Best genome  : [{}]",
        result
            .best
            .genome
            .iter()
            .take(6)
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
            .join(", ")
    );
    if result.best.genome.len() > 6 {
        println!("│               ... ({} dims total)", result.best.genome.len());
    }
    if let Some(last) = result.history.last() {
        println!("│ Final gen    : {}", last.generation);
        println!("│ Mean fitness : {:.8e}", last.mean_fitness);
        println!("│ Diversity    : {:.6}", last.diversity);
    }
    println!("└─────────────────────────────────────────────────────────\n");
}

fn main() {
    let config = EvolutionConfig {
        population_size: 200,
        max_generations: 300,
        target_fitness: Some(1e-6),
        elitism_count: 2,
        seed: Some(42),
    };

    // ── 1. Genetic Algorithm on Rastrigin ──────────────────────────
    println!("══════════════════════════════════════════════════════════");
    println!("  GENETIC ALGORITHM");
    println!("══════════════════════════════════════════════════════════\n");

    let ga = GeneticAlgorithm::default();

    let rastrigin = Rastrigin { dim: 10 };
    let result = ga.run(&rastrigin, &config);
    report("GA → Rastrigin (10D)", &result);

    let ackley = Ackley { dim: 10 };
    let result = ga.run(&ackley, &config);
    report("GA → Ackley (10D)", &result);

    // ── 2. Differential Evolution ──────────────────────────────────
    println!("══════════════════════════════════════════════════════════");
    println!("  DIFFERENTIAL EVOLUTION");
    println!("══════════════════════════════════════════════════════════\n");

    let de = DifferentialEvolution {
        strategy: DEStrategy::CurrentToBest1,
        f: 0.7,
        cr: 0.9,
        dither: true,
    };

    let rosenbrock = Rosenbrock { dim: 10 };
    let result = de.run(&rosenbrock, &config);
    report("DE/current-to-best/1 → Rosenbrock (10D)", &result);

    let schwefel = Schwefel { dim: 10 };
    let result = de.run(&schwefel, &config);
    report("DE/current-to-best/1 → Schwefel (10D)", &result);

    // ── 3. CMA-ES ──────────────────────────────────────────────────
    println!("══════════════════════════════════════════════════════════");
    println!("  CMA-ES");
    println!("══════════════════════════════════════════════════════════\n");

    let cmaes = CmaEs { initial_sigma: 0.3 };
    let levy = Levy { dim: 10 };
    let result = cmaes.run(&levy, &config);
    report("CMA-ES → Levy (10D)", &result);

    let griewank = Griewank { dim: 10 };
    let result = cmaes.run(&griewank, &config);
    report("CMA-ES → Griewank (10D)", &result);

    // ── 4. Island Model GA ─────────────────────────────────────────
    println!("══════════════════════════════════════════════════════════");
    println!("  ISLAND MODEL (4 islands, ring migration)");
    println!("══════════════════════════════════════════════════════════\n");

    let island_config = IslandModelConfig {
        num_islands: 4,
        migration_interval: 20,
        migration_count: 3,
        base_config: config.clone(),
        ..Default::default()
    };

    let rastrigin20 = Rastrigin { dim: 20 };
    let result = run_island_model(&rastrigin20, &island_config);
    report("Island Model → Rastrigin (20D)", &result);

    // ── 5. NSGA-II multi-objective ─────────────────────────────────
    println!("══════════════════════════════════════════════════════════");
    println!("  NSGA-II (Multi-Objective)");
    println!("══════════════════════════════════════════════════════════\n");

    let nsga2 = Nsga2::default();
    let zdt1 = Zdt1 { dim: 30 };
    let mo_config = EvolutionConfig {
        population_size: 200,
        max_generations: 250,
        target_fitness: None,
        elitism_count: 0,
        seed: Some(42),
    };
    let result = nsga2.run(&zdt1, &mo_config);

    println!("┌─────────────────────────────────────────────────────────");
    println!("│ NSGA-II → ZDT1 (30D, 2 objectives)");
    println!("├─────────────────────────────────────────────────────────");
    println!("│ Pareto front size: {}", result.pareto_front.len());
    println!("│ Sample Pareto solutions (f1, f2):");
    let step = (result.pareto_front.len() / 8).max(1);
    for (i, ind) in result.pareto_front.iter().enumerate().step_by(step) {
        println!(
            "│   [{:3}] f1={:.4}, f2={:.4}",
            i, ind.objectives[0], ind.objectives[1]
        );
    }
    println!("└─────────────────────────────────────────────────────────\n");

    // ── Summary ────────────────────────────────────────────────────
    println!("All benchmarks complete.");
}
