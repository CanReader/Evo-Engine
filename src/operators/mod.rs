pub mod crossover;
pub mod mutation;
pub mod selection;

use crate::Individual;
use rand::Rng;

// ---------------------------------------------------------------------------
// Operator traits - implement these to plug in custom operators
// ---------------------------------------------------------------------------

/// Crossover operator that produces two children from two parents.
pub trait CrossoverOp: Send + Sync {
    fn crossover(
        &self,
        p1: &[f64],
        p2: &[f64],
        bounds: &[(f64, f64)],
        rng: &mut dyn RngCore,
    ) -> (Vec<f64>, Vec<f64>);
}

/// Mutation operator that modifies a genome in place.
pub trait MutationOp: Send + Sync {
    fn mutate(
        &self,
        genome: &mut [f64],
        bounds: &[(f64, f64)],
        rng: &mut dyn RngCore,
    );
}

/// Selection operator that picks one individual from the population.
pub trait SelectionOp: Send + Sync {
    fn select<'a>(
        &self,
        pop: &'a [Individual<Vec<f64>>],
        rng: &mut dyn RngCore,
    ) -> &'a Individual<Vec<f64>>;
}

// We need a trait-object-safe RNG wrapper since `impl Rng` can't be
// used in trait objects. This re-exports rand_core::RngCore which is
// already object-safe.
pub use rand::RngCore;

// ---------------------------------------------------------------------------
// Built-in operator wrappers implementing the traits above
// ---------------------------------------------------------------------------

/// SBX crossover wrapped as a trait object.
pub struct SbxCrossover {
    pub eta: f64,
}

impl Default for SbxCrossover {
    fn default() -> Self {
        Self { eta: 20.0 }
    }
}

impl CrossoverOp for SbxCrossover {
    fn crossover(
        &self,
        p1: &[f64],
        p2: &[f64],
        bounds: &[(f64, f64)],
        rng: &mut dyn RngCore,
    ) -> (Vec<f64>, Vec<f64>) {
        crossover::sbx_crossover(p1, p2, bounds, self.eta, rng)
    }
}

/// BLX-alpha crossover wrapped as a trait object.
pub struct BlxAlphaCrossover {
    pub alpha: f64,
}

impl Default for BlxAlphaCrossover {
    fn default() -> Self {
        Self { alpha: 0.5 }
    }
}

impl CrossoverOp for BlxAlphaCrossover {
    fn crossover(
        &self,
        p1: &[f64],
        p2: &[f64],
        bounds: &[(f64, f64)],
        rng: &mut dyn RngCore,
    ) -> (Vec<f64>, Vec<f64>) {
        crossover::blx_alpha_crossover(p1, p2, bounds, self.alpha, rng)
    }
}

/// Arithmetic crossover wrapped as a trait object.
pub struct ArithmeticCrossoverOp;

impl CrossoverOp for ArithmeticCrossoverOp {
    fn crossover(
        &self,
        p1: &[f64],
        p2: &[f64],
        _bounds: &[(f64, f64)],
        rng: &mut dyn RngCore,
    ) -> (Vec<f64>, Vec<f64>) {
        crossover::arithmetic_crossover(p1, p2, rng)
    }
}

/// Polynomial mutation wrapped as a trait object.
pub struct PolynomialMutation {
    pub eta: f64,
    pub prob: f64,
}

impl PolynomialMutation {
    pub fn new(eta: f64, prob: f64) -> Self {
        Self { eta, prob }
    }
}

impl Default for PolynomialMutation {
    fn default() -> Self {
        Self {
            eta: 20.0,
            prob: 0.1,
        }
    }
}

impl MutationOp for PolynomialMutation {
    fn mutate(
        &self,
        genome: &mut [f64],
        bounds: &[(f64, f64)],
        rng: &mut dyn RngCore,
    ) {
        let per_gene = self.prob / genome.len().max(1) as f64;
        mutation::polynomial_mutation(genome, bounds, self.eta, per_gene, rng);
    }
}

/// Gaussian mutation wrapped as a trait object.
pub struct GaussianMutation {
    pub sigma: f64,
    pub prob: f64,
}

impl Default for GaussianMutation {
    fn default() -> Self {
        Self {
            sigma: 0.1,
            prob: 0.1,
        }
    }
}

impl MutationOp for GaussianMutation {
    fn mutate(
        &self,
        genome: &mut [f64],
        bounds: &[(f64, f64)],
        rng: &mut dyn RngCore,
    ) {
        mutation::gaussian_mutation(genome, bounds, self.sigma, self.prob, rng);
    }
}

/// Cauchy mutation wrapped as a trait object.
pub struct CauchyMutation {
    pub scale: f64,
    pub prob: f64,
}

impl Default for CauchyMutation {
    fn default() -> Self {
        Self {
            scale: 1.0,
            prob: 0.1,
        }
    }
}

impl MutationOp for CauchyMutation {
    fn mutate(
        &self,
        genome: &mut [f64],
        bounds: &[(f64, f64)],
        rng: &mut dyn RngCore,
    ) {
        mutation::cauchy_mutation(genome, bounds, self.scale, self.prob, rng);
    }
}

/// Tournament selection wrapped as a trait object.
pub struct TournamentSelection {
    pub k: usize,
}

impl Default for TournamentSelection {
    fn default() -> Self {
        Self { k: 3 }
    }
}

impl SelectionOp for TournamentSelection {
    fn select<'a>(
        &self,
        pop: &'a [Individual<Vec<f64>>],
        rng: &mut dyn RngCore,
    ) -> &'a Individual<Vec<f64>> {
        selection::tournament_select(pop, self.k, rng)
    }
}

/// Rank-based selection wrapped as a trait object.
pub struct RankSelection;

impl SelectionOp for RankSelection {
    fn select<'a>(
        &self,
        pop: &'a [Individual<Vec<f64>>],
        rng: &mut dyn RngCore,
    ) -> &'a Individual<Vec<f64>> {
        selection::rank_select(pop, rng)
    }
}
