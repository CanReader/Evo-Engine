pub mod algorithms;
pub mod island;
pub mod operators;
pub mod problems;

use ordered_float::OrderedFloat;
use std::fmt;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur during an evolutionary run.
#[derive(Debug, thiserror::Error)]
pub enum EvoError {
    #[error("population size must be at least 4, got {0}")]
    PopulationTooSmall(usize),

    #[error("dimension must be at least 1")]
    ZeroDimension,

    #[error("max_generations must be at least 1")]
    ZeroGenerations,

    #[error("elitism count ({elitism}) exceeds population size ({pop_size})")]
    ElitismTooLarge { elitism: usize, pop_size: usize },

    #[error("bounds length ({bounds}) does not match dimension ({dim})")]
    BoundsDimensionMismatch { bounds: usize, dim: usize },

    #[error("{0}")]
    Custom(String),
}

pub type EvoResult<T> = Result<T, EvoError>;

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

/// A single candidate solution in the population.
///
/// Holds the genome (decision variables), objective values, and metadata
/// used by multi-objective algorithms like NSGA-II.
#[derive(Clone, Debug)]
pub struct Individual<G: Clone> {
    pub genome: G,
    pub objectives: Vec<f64>,
    /// Domination rank assigned by non-dominated sorting (lower = better).
    pub rank: usize,
    /// Crowding distance used for diversity preservation in NSGA-II.
    pub crowding_distance: f64,
    /// Sum of constraint violations. Zero means the solution is feasible.
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

    /// Returns the first objective value. Used as "the" fitness in
    /// single-objective algorithms. Returns +inf when no objectives are set.
    pub fn fitness(&self) -> f64 {
        self.objectives.first().copied().unwrap_or(f64::INFINITY)
    }

    /// Returns true if this individual is feasible (no constraint violations).
    pub fn is_feasible(&self) -> bool {
        self.constraint_violation <= 0.0
    }

    /// Pareto dominance: `self` dominates `other` when it is at least as
    /// good on every objective and strictly better on at least one.
    /// Infeasible solutions never dominate feasible ones.
    pub fn dominates(&self, other: &Self) -> bool {
        // Feasibility-first: feasible always beats infeasible
        if self.is_feasible() != other.is_feasible() {
            return self.is_feasible();
        }
        // Both infeasible: the one with less violation dominates
        if !self.is_feasible() {
            return self.constraint_violation < other.constraint_violation;
        }
        // Both feasible: standard Pareto dominance
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

    /// Feasibility-first comparison for single-objective.
    /// Feasible beats infeasible. When both feasible, lower fitness wins.
    /// When both infeasible, lower constraint violation wins.
    pub fn constrained_cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.is_feasible() && !other.is_feasible() {
            return std::cmp::Ordering::Less;
        }
        if !self.is_feasible() && other.is_feasible() {
            return std::cmp::Ordering::Greater;
        }
        if !self.is_feasible() {
            return self
                .constraint_violation
                .partial_cmp(&other.constraint_violation)
                .unwrap_or(std::cmp::Ordering::Equal);
        }
        self.fitness()
            .partial_cmp(&other.fitness())
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl<G: Clone + fmt::Display> fmt::Display for Individual<G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Individual(obj={:?}, genome={})",
            self.objectives, self.genome
        )
    }
}

// ---------------------------------------------------------------------------
// Problem trait
// ---------------------------------------------------------------------------

/// Defines an optimization problem.
///
/// Implement this trait to plug your own objective function into any
/// algorithm provided by this crate.
pub trait Problem: Send + Sync {
    type Genome: Clone + Send + Sync;

    /// How many objectives this problem has (1 for single-objective).
    fn num_objectives(&self) -> usize;

    /// Number of decision variables.
    fn dimension(&self) -> usize;

    /// Lower and upper bounds for each decision variable.
    fn bounds(&self) -> Vec<(f64, f64)>;

    /// Evaluate a candidate solution. Fill `individual.objectives` and
    /// optionally set `individual.constraint_violation`.
    fn evaluate(&self, individual: &mut Individual<Self::Genome>);

    /// Create a random genome within bounds.
    fn random_genome(&self, rng: &mut impl rand::Rng) -> Self::Genome;
}

// ---------------------------------------------------------------------------
// Callback trait
// ---------------------------------------------------------------------------

/// Hook into the evolution loop to observe or control progress.
///
/// Implement this to add logging, progress bars, checkpointing, or
/// custom early-stopping logic.
pub trait Callback<G: Clone> {
    /// Called after each generation. Return `false` to stop the run early.
    fn on_generation(
        &mut self,
        _gen: usize,
        _stats: &GenerationStats,
        _best: &Individual<G>,
    ) -> bool {
        true
    }

    /// Called when a new best solution is found.
    fn on_improvement(&mut self, _gen: usize, _new_best: &Individual<G>) {}
}

/// A callback that does nothing. Used as the default when no callback is provided.
pub struct NoCallback;
impl<G: Clone> Callback<G> for NoCallback {}

// ---------------------------------------------------------------------------
// Generation statistics
// ---------------------------------------------------------------------------

/// Statistics collected at the end of each generation.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GenerationStats {
    pub generation: usize,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub worst_fitness: f64,
    pub diversity: f64,
}

// ---------------------------------------------------------------------------
// Evolution configuration
// ---------------------------------------------------------------------------

/// Shared configuration for all evolutionary algorithms.
#[derive(Clone, Debug)]
pub struct EvolutionConfig {
    pub population_size: usize,
    pub max_generations: usize,
    /// Stop early when best fitness drops below this threshold.
    pub target_fitness: Option<f64>,
    /// Number of best individuals carried over unchanged each generation.
    pub elitism_count: usize,
    /// Optional RNG seed for reproducibility.
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

impl EvolutionConfig {
    /// Validate config values before starting a run.
    pub fn validate(&self) -> EvoResult<()> {
        if self.population_size < 4 {
            return Err(EvoError::PopulationTooSmall(self.population_size));
        }
        if self.max_generations == 0 {
            return Err(EvoError::ZeroGenerations);
        }
        if self.elitism_count >= self.population_size {
            return Err(EvoError::ElitismTooLarge {
                elitism: self.elitism_count,
                pop_size: self.population_size,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Evolution result
// ---------------------------------------------------------------------------

/// Everything returned after an evolutionary run finishes.
#[derive(Clone, Debug)]
pub struct EvolutionResult<G: Clone> {
    /// Best individual found across all generations.
    pub best: Individual<G>,
    /// Pareto front (only populated by multi-objective algorithms).
    pub pareto_front: Vec<Individual<G>>,
    /// Per-generation statistics.
    pub history: Vec<GenerationStats>,
    /// How many generations actually ran (may be less than max if early-stopped).
    pub generations_run: usize,
}

impl EvolutionResult<Vec<f64>> {
    /// Export the generation history as CSV text.
    ///
    /// Each line has: generation, best_fitness, mean_fitness, worst_fitness, diversity
    pub fn history_to_csv(&self) -> String {
        let mut buf =
            String::from("generation,best_fitness,mean_fitness,worst_fitness,diversity\n");
        for s in &self.history {
            buf.push_str(&format!(
                "{},{},{},{},{}\n",
                s.generation, s.best_fitness, s.mean_fitness, s.worst_fitness, s.diversity
            ));
        }
        buf
    }

    /// Export the generation history as a JSON array.
    #[cfg(feature = "serialize")]
    pub fn history_to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.history)
    }

    /// Export the Pareto front as CSV (one line per solution, objectives only).
    pub fn pareto_to_csv(&self) -> String {
        if self.pareto_front.is_empty() {
            return String::new();
        }
        let num_obj = self.pareto_front[0].objectives.len();
        let header: Vec<String> = (0..num_obj).map(|i| format!("f{}", i + 1)).collect();
        let mut buf = header.join(",") + "\n";
        for ind in &self.pareto_front {
            let vals: Vec<String> = ind.objectives.iter().map(|v| v.to_string()).collect();
            buf.push_str(&vals.join(","));
            buf.push('\n');
        }
        buf
    }
}

// ---------------------------------------------------------------------------
// Algorithm trait
// ---------------------------------------------------------------------------

/// Common interface for all evolutionary algorithms.
pub trait EvolutionaryAlgorithm {
    type Genome: Clone + Send + Sync;

    fn run<P: Problem<Genome = Self::Genome>>(
        &self,
        problem: &P,
        config: &EvolutionConfig,
    ) -> EvoResult<EvolutionResult<Self::Genome>>;
}

// ---------------------------------------------------------------------------
// Utility: Euclidean diversity of a real-valued population
// ---------------------------------------------------------------------------

/// Computes the average Euclidean distance of individuals from the
/// population centroid. A simple measure of how spread out the
/// population is in decision space.
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

/// Assigns domination ranks to every individual using fast non-dominated
/// sorting. Rank 0 = the Pareto front, rank 1 = dominated only by rank 0, etc.
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

/// Assigns crowding distances within a single front. Boundary solutions
/// get infinite distance so they are always preserved.
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
