# evo-engine

[![CI](https://github.com/CanReader/Evo-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/CanReader/Evo-Engine/actions/workflows/ci.yml)

Evolutionary computation framework for Rust. Provides GA, Differential Evolution, CMA-ES, NSGA-II, and an island model out of the box, along with a rich set of genetic operators and benchmark problems.

Designed to be useful both as a ready-to-go optimizer and as a toolkit for building your own evolutionary algorithms from composable parts.

## Features

- **Multiple algorithms** - GA, DE (3 strategies), CMA-ES, NSGA-II, island model
- **Pluggable operators** - swap in any crossover, mutation, or selection via trait objects and the builder API
- **Real, binary, and permutation genomes** - continuous optimization, bitstring problems, and routing/scheduling
- **Constraint handling** - feasibility-first comparison baked into all algorithms
- **Benchmark suite** - Rastrigin, Rosenbrock, Ackley, Schwefel, Griewank, Levy, ZDT1-3, OneMax, TSP, Knapsack
- **CSV/JSON export** - dump generation history for plotting or analysis
- **Callbacks** - hook into the evolution loop for logging, checkpointing, or custom stopping
- **Feature flags** - `parallel` (rayon) and `serialize` (serde) are optional, both on by default
- **Tested** - 30 unit and integration tests covering operators, algorithms, and edge cases

## Quick start

```bash
# Run the built-in benchmark suite
cargo run --release

# Run an example
cargo run --example custom_problem
cargo run --example tsp
cargo run --example knapsack
cargo run --example builder_api
cargo run --example csv_export
```

### Basic usage

```rust
use evo_engine::algorithms::ga::GeneticAlgorithm;
use evo_engine::problems::single_objective::Rastrigin;
use evo_engine::{EvolutionConfig, EvolutionaryAlgorithm};

fn main() {
    let ga = GeneticAlgorithm::default();
    let problem = Rastrigin { dim: 10 };
    let config = EvolutionConfig {
        population_size: 200,
        max_generations: 300,
        target_fitness: Some(1e-6),
        elitism_count: 2,
        seed: Some(42),
    };

    let result = ga.run(&problem, &config).expect("run failed");
    println!("Best fitness: {:.6e}", result.best.fitness());
}
```

### Custom operators via builder

```rust
use evo_engine::algorithms::ga::GeneticAlgorithmBuilder;
use evo_engine::operators::{BlxAlphaCrossover, GaussianMutation, TournamentSelection};

let ga = GeneticAlgorithmBuilder::new()
    .crossover(BlxAlphaCrossover { alpha: 0.5 })
    .mutation(GaussianMutation { sigma: 0.1, prob: 0.05 })
    .selection(TournamentSelection { k: 5 })
    .crossover_prob(0.85)
    .build();
```

### Exporting results

```rust
// CSV (pipe to file or parse in Python/R)
let csv = result.history_to_csv();
std::fs::write("history.csv", &csv).unwrap();

// JSON (requires "serialize" feature)
let json = result.history_to_json().unwrap();

// Pareto front as CSV
let pareto_csv = result.pareto_to_csv();
```

## Algorithms

| Algorithm | Description | Best for |
|---|---|---|
| **GA** | Tournament selection + configurable crossover/mutation | General-purpose single-objective |
| **DE** | Difference-vector mutation (rand/1, best/1, current-to-best/1) | Continuous parameter tuning |
| **CMA-ES** | Diagonal covariance adaptation with step-size control | Ill-conditioned landscapes |
| **NSGA-II** | Non-dominated sorting + crowding distance | Multi-objective optimization |
| **Island Model** | Multiple sub-populations with ring migration | Keeping diversity on hard problems |

## Operators

### Crossover (real-valued)
- **SBX** - Simulated Binary Crossover with distribution index eta
- **BLX-alpha** - blend crossover with exploration parameter
- **Arithmetic** - linear combination of parents
- **UNDX** - 3-parent unimodal normal distribution crossover

### Crossover (binary)
- **Single-point**, **two-point**, **uniform**

### Crossover (permutation)
- **OX** - order crossover
- **PMX** - partially-mapped crossover

### Mutation
- **Polynomial** - bounded, pairs well with SBX
- **Gaussian** - normal perturbation
- **Cauchy** - heavy-tailed, good for escaping local optima
- **Adaptive** - self-adapting step-size (1/5 success rule)
- **Non-uniform** - perturbation decreases over generations
- **Bitflip** - for binary genomes
- **Swap, insert, inversion** - for permutation genomes

### Selection
- **Tournament** - k-way tournament
- **Crowded tournament** - NSGA-II rank + crowding distance
- **SUS** - stochastic universal sampling
- **Rank-based** - linear ranking

## Project structure

```
src/
  lib.rs                          Core traits, error types, callbacks, CSV export
  main.rs                         Benchmark runner
  algorithms/
    ga.rs                         Genetic Algorithm with builder API
    differential_evolution.rs     DE with 3 strategy variants
    cmaes.rs                      CMA-ES (diagonal covariance)
    nsga2.rs                      NSGA-II multi-objective
  operators/
    mod.rs                        Operator traits + built-in wrappers
    crossover.rs                  SBX, BLX-alpha, arithmetic, UNDX
    mutation.rs                   Polynomial, Gaussian, Cauchy, adaptive, non-uniform
    selection.rs                  Tournament, crowded tournament, SUS, rank-based
    binary.rs                     Single-point, two-point, uniform crossover; bitflip mutation
    permutation.rs                OX, PMX crossover; swap, insert, inversion mutation
  problems/
    single_objective.rs           Rastrigin, Rosenbrock, Ackley, Schwefel, Griewank, Levy
    multi_objective.rs            ZDT1, ZDT2, ZDT3
    combinatorial.rs              OneMax, TSP, Knapsack
  island/
    mod.rs                        Parallel island model with ring migration
examples/
  custom_problem.rs               Define and solve your own problem
  builder_api.rs                  GA with custom operators
  tsp.rs                          Traveling Salesman with permutation operators
  knapsack.rs                     0/1 Knapsack with binary operators
  csv_export.rs                   Export history to CSV/JSON
tests/
  operators.rs                    Unit tests for all operator types
  algorithms.rs                   Integration tests for every algorithm
```

## Feature flags

| Feature | Default | What it enables |
|---|---|---|
| `parallel` | yes | Rayon-based parallelism for the island model |
| `serialize` | yes | Serde support for `GenerationStats`, JSON export |

Build without optional deps:

```bash
cargo build --no-default-features
```

## Requirements

- Rust 2021 edition (1.56+)
- No C/C++ dependencies, pure Rust

## License

MIT
