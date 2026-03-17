//! Operators for permutation (Vec<usize>) genomes.
//!
//! These are designed for problems like TSP where the genome is an
//! ordering of items and standard crossover would create invalid solutions.

use rand::{Rng, RngCore};

/// Order crossover (OX). Copies a random slice from p1 directly, then
/// fills the remaining positions with elements from p2 in the order
/// they appear, skipping any that are already placed.
pub fn order_crossover(
    p1: &[usize],
    p2: &[usize],
    rng: &mut dyn RngCore,
) -> Vec<usize> {
    let n = p1.len();
    let mut a = rng.gen_range(0..n);
    let mut b = rng.gen_range(0..n);
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }

    let mut child = vec![usize::MAX; n];
    // Copy the slice from p1
    for i in a..=b {
        child[i] = p1[i];
    }

    // Fill remaining from p2, preserving p2's relative order
    let mut pos = (b + 1) % n;
    for &gene in p2.iter().cycle().skip(b + 1).take(n) {
        if !child.contains(&gene) {
            child[pos] = gene;
            pos = (pos + 1) % n;
        }
    }
    child
}

/// Partially-mapped crossover (PMX). Establishes a mapping between
/// corresponding elements in a slice, then uses that mapping to
/// resolve conflicts when merging the two parents.
pub fn pmx_crossover(
    p1: &[usize],
    p2: &[usize],
    rng: &mut dyn RngCore,
) -> (Vec<usize>, Vec<usize>) {
    let n = p1.len();
    let mut a = rng.gen_range(0..n);
    let mut b = rng.gen_range(0..n);
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }

    fn build_child(donor: &[usize], filler: &[usize], a: usize, b: usize) -> Vec<usize> {
        let n = donor.len();
        let mut child = vec![usize::MAX; n];

        // Copy the segment from the donor
        for i in a..=b {
            child[i] = donor[i];
        }

        // Try to place filler elements
        for i in a..=b {
            if child.contains(&filler[i]) {
                continue;
            }
            // Follow the mapping chain until we find a free position
            let val = filler[i];
            let mut target = i;
            loop {
                // Where does donor[target] sit in filler?
                let pos = filler.iter().position(|&x| x == donor[target]).unwrap();
                if pos < a || pos > b {
                    child[pos] = val;
                    break;
                }
                target = pos;
            }
        }

        // Fill any remaining slots from filler
        for i in 0..n {
            if child[i] == usize::MAX {
                child[i] = filler[i];
            }
        }
        child
    }

    let c1 = build_child(p1, p2, a, b);
    let c2 = build_child(p2, p1, a, b);
    (c1, c2)
}

/// Swap mutation. Picks two random positions and swaps them.
pub fn swap_mutation(genome: &mut [usize], rng: &mut dyn RngCore) {
    let n = genome.len();
    if n < 2 {
        return;
    }
    let i = rng.gen_range(0..n);
    let j = rng.gen_range(0..n);
    genome.swap(i, j);
}

/// Insert mutation. Removes an element from one position and inserts
/// it at another, shifting everything in between.
pub fn insert_mutation(genome: &mut Vec<usize>, rng: &mut dyn RngCore) {
    let n = genome.len();
    if n < 2 {
        return;
    }
    let from = rng.gen_range(0..n);
    let to = rng.gen_range(0..n);
    let val = genome.remove(from);
    genome.insert(to, val);
}

/// Inversion mutation. Reverses the sub-sequence between two random points.
/// This is a gentle operator for TSP since reversing a segment preserves
/// most neighbor relationships.
pub fn inversion_mutation(genome: &mut [usize], rng: &mut dyn RngCore) {
    let n = genome.len();
    if n < 2 {
        return;
    }
    let mut a = rng.gen_range(0..n);
    let mut b = rng.gen_range(0..n);
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    genome[a..=b].reverse();
}
