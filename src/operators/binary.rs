//! Operators for binary (Vec<bool>) genomes.

use rand::{Rng, RngCore};

/// Single-point crossover. Splits both parents at a random point
/// and swaps the tails.
pub fn single_point_crossover(
    p1: &[bool],
    p2: &[bool],
    rng: &mut dyn RngCore,
) -> (Vec<bool>, Vec<bool>) {
    let n = p1.len();
    let point = rng.gen_range(1..n);
    let c1 = [&p1[..point], &p2[point..]].concat();
    let c2 = [&p2[..point], &p1[point..]].concat();
    (c1, c2)
}

/// Two-point crossover. Swaps the segment between two random points.
pub fn two_point_crossover(
    p1: &[bool],
    p2: &[bool],
    rng: &mut dyn RngCore,
) -> (Vec<bool>, Vec<bool>) {
    let n = p1.len();
    let mut a = rng.gen_range(0..n);
    let mut b = rng.gen_range(0..n);
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }

    let mut c1 = p1.to_vec();
    let mut c2 = p2.to_vec();
    for i in a..=b {
        c1[i] = p2[i];
        c2[i] = p1[i];
    }
    (c1, c2)
}

/// Uniform crossover. Each bit is independently taken from one parent
/// or the other with equal probability.
pub fn uniform_crossover(
    p1: &[bool],
    p2: &[bool],
    rng: &mut dyn RngCore,
) -> (Vec<bool>, Vec<bool>) {
    let n = p1.len();
    let mut c1 = Vec::with_capacity(n);
    let mut c2 = Vec::with_capacity(n);
    for i in 0..n {
        if rng.gen_bool(0.5) {
            c1.push(p1[i]);
            c2.push(p2[i]);
        } else {
            c1.push(p2[i]);
            c2.push(p1[i]);
        }
    }
    (c1, c2)
}

/// Bit-flip mutation. Each bit is flipped independently with the given
/// probability. A common default is 1/n where n is the genome length.
pub fn bitflip_mutation(
    genome: &mut [bool],
    prob: f64,
    rng: &mut dyn RngCore,
) {
    for bit in genome.iter_mut() {
        if rng.gen::<f64>() < prob {
            *bit = !*bit;
        }
    }
}
