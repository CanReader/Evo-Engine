use evo_engine::algorithms::cmaes::CmaEs;
use evo_engine::algorithms::differential_evolution::{DEStrategy, DifferentialEvolution};
use evo_engine::algorithms::ga::GeneticAlgorithm;
use evo_engine::algorithms::nsga2::Nsga2;
use evo_engine::island::{run_island_model, IslandModelConfig};
use evo_engine::problems::multi_objective::Zdt1;
use evo_engine::problems::single_objective::{Rastrigin, Rosenbrock, Ackley};
use evo_engine::{EvolutionConfig, EvolutionaryAlgorithm};

fn small_config() -> EvolutionConfig {
    EvolutionConfig {
        population_size: 50,
        max_generations: 100,
        target_fitness: None,
        elitism_count: 2,
        seed: Some(42),
    }
}

// ---------------------------------------------------------------------------
// GA tests
// ---------------------------------------------------------------------------

#[test]
fn ga_converges_on_rastrigin() {
    let ga = GeneticAlgorithm::default();
    let problem = Rastrigin { dim: 5 };
    let config = EvolutionConfig {
        population_size: 100,
        max_generations: 200,
        target_fitness: Some(1.0),
        elitism_count: 2,
        seed: Some(42),
    };
    let result = ga.run(&problem, &config).unwrap();
    assert!(
        result.best.fitness() < 50.0,
        "GA should make some progress on Rastrigin, got {}",
        result.best.fitness()
    );
}

#[test]
fn ga_returns_history() {
    let ga = GeneticAlgorithm::default();
    let problem = Ackley { dim: 3 };
    let result = ga.run(&problem, &small_config()).unwrap();
    assert!(!result.history.is_empty(), "history should not be empty");
    assert!(result.generations_run > 0);
}

#[test]
fn ga_validates_config() {
    let ga = GeneticAlgorithm::default();
    let problem = Rastrigin { dim: 2 };
    let bad_config = EvolutionConfig {
        population_size: 2, // too small
        ..small_config()
    };
    assert!(ga.run(&problem, &bad_config).is_err());
}

// ---------------------------------------------------------------------------
// DE tests
// ---------------------------------------------------------------------------

#[test]
fn de_converges_on_rosenbrock() {
    let de = DifferentialEvolution {
        strategy: DEStrategy::CurrentToBest1,
        f: 0.7,
        cr: 0.9,
        dither: true,
    };
    let problem = Rosenbrock { dim: 5 };
    let config = EvolutionConfig {
        population_size: 80,
        max_generations: 300,
        target_fitness: None,
        elitism_count: 0,
        seed: Some(42),
    };
    let result = de.run(&problem, &config).unwrap();
    assert!(
        result.best.fitness() < 1000.0,
        "DE should make progress on Rosenbrock, got {}",
        result.best.fitness()
    );
}

#[test]
fn de_strategies_all_work() {
    let problem = Rastrigin { dim: 3 };
    let config = small_config();

    for strategy in [DEStrategy::Rand1Bin, DEStrategy::Best1Bin, DEStrategy::CurrentToBest1] {
        let de = DifferentialEvolution {
            strategy,
            f: 0.8,
            cr: 0.9,
            dither: false,
        };
        let result = de.run(&problem, &config);
        assert!(result.is_ok(), "DE strategy {:?} failed", strategy);
    }
}

// ---------------------------------------------------------------------------
// CMA-ES tests
// ---------------------------------------------------------------------------

#[test]
fn cmaes_converges_on_ackley() {
    let cmaes = CmaEs { initial_sigma: 0.3 };
    let problem = Ackley { dim: 5 };
    let config = EvolutionConfig {
        population_size: 50,
        max_generations: 200,
        target_fitness: None,
        elitism_count: 0,
        seed: Some(42),
    };
    let result = cmaes.run(&problem, &config).unwrap();
    assert!(
        result.best.fitness() < 10.0,
        "CMA-ES should do well on Ackley, got {}",
        result.best.fitness()
    );
}

// ---------------------------------------------------------------------------
// NSGA-II tests
// ---------------------------------------------------------------------------

#[test]
fn nsga2_produces_pareto_front() {
    let nsga2 = Nsga2::default();
    let problem = Zdt1 { dim: 10 };
    let config = EvolutionConfig {
        population_size: 50,
        max_generations: 50,
        target_fitness: None,
        elitism_count: 0,
        seed: Some(42),
    };
    let result = nsga2.run(&problem, &config).unwrap();
    assert!(
        !result.pareto_front.is_empty(),
        "NSGA-II should produce a Pareto front"
    );
    // All Pareto front members should have rank 0
    for ind in &result.pareto_front {
        assert_eq!(ind.rank, 0);
    }
}

// ---------------------------------------------------------------------------
// Island model tests
// ---------------------------------------------------------------------------

#[test]
fn island_model_runs_and_converges() {
    let problem = Rastrigin { dim: 5 };
    let config = IslandModelConfig {
        num_islands: 2,
        migration_interval: 10,
        migration_count: 2,
        base_config: EvolutionConfig {
            population_size: 40,
            max_generations: 50,
            target_fitness: None,
            elitism_count: 2,
            seed: Some(42),
        },
        ..Default::default()
    };
    let result = run_island_model(&problem, &config).unwrap();
    assert!(!result.history.is_empty());
    assert!(result.best.fitness().is_finite());
}

// ---------------------------------------------------------------------------
// CSV/JSON export tests
// ---------------------------------------------------------------------------

#[test]
fn csv_export_has_header_and_rows() {
    let ga = GeneticAlgorithm::default();
    let problem = Rastrigin { dim: 2 };
    let result = ga.run(&problem, &small_config()).unwrap();

    let csv = result.history_to_csv();
    let lines: Vec<&str> = csv.lines().collect();
    assert!(lines.len() > 1, "CSV should have header + data rows");
    assert!(lines[0].contains("generation"));
    assert!(lines[0].contains("best_fitness"));
}

#[test]
fn pareto_csv_works() {
    let nsga2 = Nsga2::default();
    let problem = Zdt1 { dim: 10 };
    let config = EvolutionConfig {
        population_size: 30,
        max_generations: 20,
        target_fitness: None,
        elitism_count: 0,
        seed: Some(42),
    };
    let result = nsga2.run(&problem, &config).unwrap();
    let csv = result.pareto_to_csv();
    assert!(!csv.is_empty());
    assert!(csv.starts_with("f1,f2"));
}

// ---------------------------------------------------------------------------
// Config validation tests
// ---------------------------------------------------------------------------

#[test]
fn config_rejects_zero_generations() {
    let config = EvolutionConfig {
        max_generations: 0,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn config_rejects_elitism_too_large() {
    let config = EvolutionConfig {
        population_size: 10,
        elitism_count: 10,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn default_config_is_valid() {
    let config = EvolutionConfig::default();
    assert!(config.validate().is_ok());
}
