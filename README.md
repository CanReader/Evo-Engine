# evo-engine

An advanced evolutionary computation framework written in Rust. Solve single-objective and multi-objective optimization problems using a variety of metaheuristic algorithms inspired by natural selection and biological evolution.

## Features

- **Multiple algorithms** &mdash; Genetic Algorithm, Differential Evolution, CMA-ES, NSGA-II
- **Island model** &mdash; parallel evolution with ring-topology migration via Rayon
- **Rich operator library** &mdash; SBX, BLX-&alpha;, UNDX, polynomial mutation, Cauchy mutation, adaptive mutation, tournament selection, SUS, and more
- **Benchmark suite** &mdash; Rastrigin, Rosenbrock, Ackley, Schwefel, Griewank, Levy, ZDT1-3
- **Trait-based design** &mdash; plug in custom problems and operators with ease
- **Serde support** &mdash; serialize generation statistics to JSON

## Algorithms

| Algorithm | Key idea | Use case |
|---|---|---|
| **GA** | Tournament selection + SBX crossover + polynomial mutation | General-purpose single-objective |
| **DE** | Difference-vector perturbation (rand/1, best/1, current-to-best/1) | Continuous parameter optimization |
| **CMA-ES** | Covariance matrix adaptation with step-size control | High-dimensional, ill-conditioned landscapes |
| **NSGA-II** | Non-dominated sorting + crowding distance | Multi-objective optimization |
| **Island Model** | Multiple sub-populations with periodic migration | Large-scale parallelism, diversity preservation |

## Quick start

```bash
# Run the built-in benchmark suite
cargo run --release

# Use as a library
cargo add evo-engine   # (or add to Cargo.toml manually)
```

### Library usage

```rust
use evo_engine::algorithms::ga::GeneticAlgorithm;
use evo_engine::problems::single_objective::Rastrigin;
use evo_engine::{EvolutionConfig, EvolutionaryAlgorithm};

fn main() {
    let config = EvolutionConfig {
        population_size: 200,
        max_generations: 300,
        target_fitness: Some(1e-6),
        elitism_count: 2,
        seed: Some(42),
    };

    let ga = GeneticAlgorithm::default();
    let problem = Rastrigin { dim: 10 };
    let result = ga.run(&problem, &config);

    println!("Best fitness: {:.6e}", result.best.fitness());
}
```

## Project structure

```
src/
  lib.rs                          Core traits, types, NSGA-II utilities
  main.rs                         Benchmark demonstration runner
  algorithms/
    ga.rs                         Genetic Algorithm
    differential_evolution.rs     Differential Evolution (3 strategies)
    cmaes.rs                      CMA-ES (diagonal covariance)
    nsga2.rs                      NSGA-II multi-objective
  operators/
    crossover.rs                  SBX, BLX-alpha, arithmetic, UNDX
    mutation.rs                   Polynomial, Gaussian, Cauchy, adaptive, non-uniform
    selection.rs                  Tournament, crowded tournament, SUS, rank-based
  problems/
    single_objective.rs           Rastrigin, Rosenbrock, Ackley, Schwefel, Griewank, Levy
    multi_objective.rs            ZDT1, ZDT2, ZDT3
  island/
    mod.rs                        Parallel island model with ring migration
```

## Operators

### Crossover
- **SBX** (Simulated Binary Crossover) &mdash; distribution index &eta;
- **BLX-&alpha;** &mdash; blend crossover with exploration parameter
- **Arithmetic** &mdash; linear combination of parents
- **UNDX** &mdash; 3-parent unimodal normal distribution crossover

### Mutation
- **Polynomial** &mdash; bounded mutation with distribution index
- **Gaussian** &mdash; normal perturbation
- **Cauchy** &mdash; heavy-tailed for escaping local optima
- **Adaptive** &mdash; self-adapting step-size via 1/5 success rule
- **Non-uniform** &mdash; generation-dependent decreasing perturbation

### Selection
- **Tournament** &mdash; k-way tournament selection
- **Crowded tournament** &mdash; NSGA-II rank + crowding distance
- **SUS** &mdash; stochastic universal sampling
- **Rank-based** &mdash; linear ranking selection

## Benchmark problems

### Single-objective
| Problem | Optimum | Characteristics |
|---|---|---|
| Rastrigin | f(0) = 0 | Highly multimodal |
| Rosenbrock | f(1) = 0 | Narrow valley, unimodal |
| Ackley | f(0) = 0 | Multimodal, nearly flat outer region |
| Schwefel | f(420.97) &asymp; 0 | Deceptive, distant global optimum |
| Griewank | f(0) = 0 | Multimodal with product term |
| Levy | f(1) = 0 | Complex landscape |

### Multi-objective (ZDT suite)
| Problem | Pareto front |
|---|---|
| ZDT1 | Convex |
| ZDT2 | Non-convex |
| ZDT3 | Disconnected |

## Requirements

- Rust 2021 edition (1.56+)
- Dependencies: `rand`, `rand_distr`, `rayon`, `ordered-float`, `serde`, `serde_json`

## License

MIT
