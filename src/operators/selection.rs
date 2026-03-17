use crate::Individual;
use rand::Rng;

// ---------------------------------------------------------------------------
// Tournament selection
// ---------------------------------------------------------------------------

pub fn tournament_select<'a, G: Clone>(
    pop: &'a [Individual<G>],
    k: usize,
    rng: &mut impl Rng,
) -> &'a Individual<G> {
    let mut best_idx = rng.gen_range(0..pop.len());
    for _ in 1..k {
        let idx = rng.gen_range(0..pop.len());
        if pop[idx].fitness() < pop[best_idx].fitness() {
            best_idx = idx;
        }
    }
    &pop[best_idx]
}

// ---------------------------------------------------------------------------
// NSGA-II crowded tournament selection
// ---------------------------------------------------------------------------

pub fn crowded_tournament_select<'a, G: Clone>(
    pop: &'a [Individual<G>],
    rng: &mut impl Rng,
) -> &'a Individual<G> {
    let a = rng.gen_range(0..pop.len());
    let b = rng.gen_range(0..pop.len());
    if pop[a].rank < pop[b].rank {
        &pop[a]
    } else if pop[b].rank < pop[a].rank {
        &pop[b]
    } else if pop[a].crowding_distance > pop[b].crowding_distance {
        &pop[a]
    } else {
        &pop[b]
    }
}

// ---------------------------------------------------------------------------
// Stochastic Universal Sampling (SUS)
// ---------------------------------------------------------------------------

pub fn stochastic_universal_sampling<G: Clone>(
    pop: &[Individual<G>],
    n: usize,
    rng: &mut impl Rng,
) -> Vec<Individual<G>> {
    // Fitness proportionate — invert because we minimise.
    let max_fit = pop.iter().map(|i| i.fitness()).fold(f64::NEG_INFINITY, f64::max);
    let fitnesses: Vec<f64> = pop.iter().map(|i| max_fit - i.fitness() + 1e-6).collect();
    let total: f64 = fitnesses.iter().sum();

    let spacing = total / n as f64;
    let start: f64 = rng.gen::<f64>() * spacing;

    let mut selected = Vec::with_capacity(n);
    let mut cumulative = 0.0;
    let mut idx = 0;
    for i in 0..n {
        let pointer = start + i as f64 * spacing;
        while cumulative + fitnesses[idx] < pointer {
            cumulative += fitnesses[idx];
            idx += 1;
        }
        selected.push(pop[idx].clone());
    }
    selected
}

// ---------------------------------------------------------------------------
// Rank-based selection
// ---------------------------------------------------------------------------

pub fn rank_select<G: Clone>(
    pop: &[Individual<G>],
    rng: &mut impl Rng,
) -> &Individual<G> {
    let n = pop.len();
    // Sort indices by fitness (ascending = best first for minimisation).
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| pop[a].fitness().partial_cmp(&pop[b].fitness()).unwrap());

    // Linear ranking: best gets rank n, worst gets rank 1.
    let total_rank: usize = n * (n + 1) / 2;
    let pick = rng.gen_range(0..total_rank);
    let mut cumulative = 0;
    for (rank_minus_1, &idx) in indices.iter().rev().enumerate() {
        cumulative += rank_minus_1 + 1;
        if cumulative > pick {
            return &pop[idx];
        }
    }
    &pop[indices[0]]
}
