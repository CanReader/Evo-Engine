pub mod algorithms;
pub mod island;
pub mod operators;
pub mod problems;

use ordered_float::OrderedFloat;
use std::fmt;

// ---------------------------------------------------------------------------
// Core traits — the backbone of the framework
// ---------------------------------------------------------------------------

/// A single candidate solution.
#[derive(Clone, Debug)]
pub struct Individual<G: Clone> {
    pub genome: G,
    pub objectives: Vec<f64>,
    /// NSGA-II: domination rank (lower is better).
    pub rank: usize,
    /// NSGA-II: crowding distance.
    pub crowding_distance: f64,
    /// Constraint violation (0.0 = feasible).
    pub constraint_violation: f64,
}

impl<G: Clone> Individual<G> {
    pub fn new(genome: G) -> Self {
        Self {
            genome,
            objectives: Vec::new(),
            rank: usize::MAX,
            crowding_distance: 0.0,
            constraint_violation: 0.0,
        }
    }

    /// Single-objective fitness (first objective, minimisation).
    pub fn fitness(&self) -> f64 {
        self.objectives.first().copied().unwrap_or(f64::INFINITY)
    }

    /// NSGA-II dominance: `self` dominates `other` if at least as good on all
    /// objectives and strictly better on at least one.
    pub fn dominates(&self, other: &Self) -> bool {
        if self.objectives.len() != other.objectives.len() {
            return false;
        }
        let mut dominated_one = false;
        for (a, b) in self.objectives.iter().zip(&other.objectives) {
            if a > b {
                return false;
            }
            if a < b {
                dominated_one = true;
            }
        }
        dominated_one
    }
}

impl<G: Clone + fmt::Display> fmt::Display for Individual<G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Individual(obj={:?}, genome={})", self.objectives, self.genome)
    }
}

/// Objective function (problem definition).
pub trait Problem: Send + Sync {
    type Genome: Clone + Send + Sync;

    /// Number of objectives (1 = single-objective).
    fn num_objectives(&self) -> usize;

    /// Dimensionality (for real-valued genomes).
    fn dimension(&self) -> usize;

    /// Bounds per dimension.
    fn bounds(&self) -> Vec<(f64, f64)>;

    /// Evaluate an individual, filling `objectives` (and optionally `constraint_violation`).
    fn evaluate(&self, individual: &mut Individual<Self::Genome>);

    /// Generate a random genome.
    fn random_genome(&self, rng: &mut impl rand::Rng) -> Self::Genome;
}

/// Statistics collected every generation.
#[derive(Clone, Debug, serde::Serialize)]
pub struct GenerationStats {
    pub generation: usize,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub worst_fitness: f64,
    pub diversity: f64,
}

/// Configuration shared across algorithms.
#[derive(Clone, Debug)]
pub struct EvolutionConfig {
    pub population_size: usize,
    pub max_generations: usize,
    pub target_fitness: Option<f64>,
    pub elitism_count: usize,
    pub seed: Option<u64>,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 200,
            max_generations: 500,
            target_fitness: None,
            elitism_count: 2,
            seed: None,
        }
    }
}

/// Result returned after an evolutionary run.
#[derive(Clone, Debug)]
pub struct EvolutionResult<G: Clone> {
    pub best: Individual<G>,
    pub pareto_front: Vec<Individual<G>>,
    pub history: Vec<GenerationStats>,
    pub generations_run: usize,
}

/// Trait for an evolutionary algorithm.
pub trait EvolutionaryAlgorithm {
    type Genome: Clone + Send + Sync;

    fn run<P: Problem<Genome = Self::Genome>>(
        &self,
        problem: &P,
        config: &EvolutionConfig,
    ) -> EvolutionResult<Self::Genome>;
}

// ---------------------------------------------------------------------------
// Utility: compute Euclidean diversity of a real-valued population
// ---------------------------------------------------------------------------

pub fn population_diversity(pop: &[Individual<Vec<f64>>]) -> f64 {
    if pop.len() < 2 {
        return 0.0;
    }
    let dim = pop[0].genome.len();
    let mean: Vec<f64> = (0..dim)
        .map(|d| pop.iter().map(|ind| ind.genome[d]).sum::<f64>() / pop.len() as f64)
        .collect();
    let var: f64 = pop
        .iter()
        .map(|ind| {
            ind.genome
                .iter()
                .zip(&mean)
                .map(|(x, m)| (x - m).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        / pop.len() as f64;
    var.sqrt()
}

// ---------------------------------------------------------------------------
// Non-dominated sorting (NSGA-II)
// ---------------------------------------------------------------------------

pub fn non_dominated_sort<G: Clone>(pop: &mut [Individual<G>]) {
    let n = pop.len();
    let mut domination_count = vec![0usize; n];
    let mut dominated_set: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut fronts: Vec<Vec<usize>> = vec![Vec::new()];

    for i in 0..n {
        for j in (i + 1)..n {
            if pop[i].dominates(&pop[j]) {
                dominated_set[i].push(j);
                domination_count[j] += 1;
            } else if pop[j].dominates(&pop[i]) {
                dominated_set[j].push(i);
                domination_count[i] += 1;
            }
        }
        if domination_count[i] == 0 {
            pop[i].rank = 0;
            fronts[0].push(i);
        }
    }

    let mut current_front = 0;
    while !fronts[current_front].is_empty() {
        let mut next_front = Vec::new();
        for &i in &fronts[current_front] {
            for &j in &dominated_set[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    pop[j].rank = current_front + 1;
                    next_front.push(j);
                }
            }
        }
        current_front += 1;
        fronts.push(next_front);
    }
}

pub fn crowding_distance_assignment<G: Clone>(pop: &mut [Individual<G>], front: &[usize]) {
    let n = front.len();
    if n <= 2 {
        for &i in front {
            pop[i].crowding_distance = f64::INFINITY;
        }
        return;
    }
    let num_obj = pop[front[0]].objectives.len();
    for &i in front {
        pop[i].crowding_distance = 0.0;
    }

    for m in 0..num_obj {
        let mut sorted_front: Vec<usize> = front.to_vec();
        sorted_front.sort_by_key(|&i| OrderedFloat(pop[i].objectives[m]));

        pop[sorted_front[0]].crowding_distance = f64::INFINITY;
        pop[sorted_front[n - 1]].crowding_distance = f64::INFINITY;

        let f_min = pop[sorted_front[0]].objectives[m];
        let f_max = pop[sorted_front[n - 1]].objectives[m];
        let span = f_max - f_min;
        if span < 1e-30 {
            continue;
        }

        for k in 1..(n - 1) {
            let dist = (pop[sorted_front[k + 1]].objectives[m]
                - pop[sorted_front[k - 1]].objectives[m])
                / span;
            pop[sorted_front[k]].crowding_distance += dist;
        }
    }
}
