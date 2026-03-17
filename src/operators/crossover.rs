use rand::{Rng, RngCore};
use rand_distr::{Distribution, Normal};

// ---------------------------------------------------------------------------
// Simulated Binary Crossover (SBX)
// ---------------------------------------------------------------------------

/// SBX crossover with distribution index `eta`. Higher eta means children
/// stay closer to parents. eta=20 is a common default.
pub fn sbx_crossover(
    p1: &[f64],
    p2: &[f64],
    bounds: &[(f64, f64)],
    eta: f64,
    rng: &mut dyn RngCore,
) -> (Vec<f64>, Vec<f64>) {
    let n = p1.len();
    let mut c1 = vec![0.0; n];
    let mut c2 = vec![0.0; n];

    for i in 0..n {
        if rng.gen::<f64>() > 0.5 || (p1[i] - p2[i]).abs() < 1e-14 {
            c1[i] = p1[i];
            c2[i] = p2[i];
            continue;
        }

        let (lo, hi) = bounds[i];
        let y1 = p1[i].min(p2[i]);
        let y2 = p1[i].max(p2[i]);
        let diff = y2 - y1;

        let beta = |yl: f64, yu: f64| -> f64 {
            let beta1 = 1.0 + 2.0 * (yl - lo) / diff;
            let beta2 = 1.0 + 2.0 * (hi - yu) / diff;
            let alpha1 = 2.0 - beta1.powf(-(eta + 1.0));
            let alpha2 = 2.0 - beta2.powf(-(eta + 1.0));
            let u: f64 = rng.gen();
            let bq = |alpha: f64| -> f64 {
                if u <= 1.0 / alpha {
                    (u * alpha).powf(1.0 / (eta + 1.0))
                } else {
                    (1.0 / (2.0 - u * alpha)).powf(1.0 / (eta + 1.0))
                }
            };
            (bq(alpha1) + bq(alpha2)) / 2.0
        };

        let beta_q = beta(y1, y2);
        c1[i] = (0.5 * ((y1 + y2) - beta_q * diff)).clamp(lo, hi);
        c2[i] = (0.5 * ((y1 + y2) + beta_q * diff)).clamp(lo, hi);
    }
    (c1, c2)
}

// ---------------------------------------------------------------------------
// BLX-alpha crossover (Blend crossover)
// ---------------------------------------------------------------------------

/// Blend crossover. Children are sampled uniformly from an interval
/// that extends `alpha` beyond each parent. alpha=0.5 is typical.
pub fn blx_alpha_crossover(
    p1: &[f64],
    p2: &[f64],
    bounds: &[(f64, f64)],
    alpha: f64,
    rng: &mut dyn RngCore,
) -> (Vec<f64>, Vec<f64>) {
    let n = p1.len();
    let mut c1 = vec![0.0; n];
    let mut c2 = vec![0.0; n];

    for i in 0..n {
        let lo = p1[i].min(p2[i]);
        let hi = p1[i].max(p2[i]);
        let range = hi - lo;
        let min_val = lo - alpha * range;
        let max_val = hi + alpha * range;
        c1[i] = (rng.gen::<f64>() * (max_val - min_val) + min_val).clamp(bounds[i].0, bounds[i].1);
        c2[i] = (rng.gen::<f64>() * (max_val - min_val) + min_val).clamp(bounds[i].0, bounds[i].1);
    }
    (c1, c2)
}

// ---------------------------------------------------------------------------
// Arithmetic (whole/uniform) crossover
// ---------------------------------------------------------------------------

/// Simple linear combination of two parents with a random mixing weight.
pub fn arithmetic_crossover(
    p1: &[f64],
    p2: &[f64],
    rng: &mut dyn RngCore,
) -> (Vec<f64>, Vec<f64>) {
    let alpha: f64 = rng.gen();
    let c1 = p1
        .iter()
        .zip(p2)
        .map(|(a, b)| alpha * a + (1.0 - alpha) * b)
        .collect();
    let c2 = p1
        .iter()
        .zip(p2)
        .map(|(a, b)| (1.0 - alpha) * a + alpha * b)
        .collect();
    (c1, c2)
}

// ---------------------------------------------------------------------------
// UNDX (Unimodal Normal Distribution Crossover) - 3-parent
// ---------------------------------------------------------------------------

/// Three-parent crossover using a Gaussian distribution centered between
/// two parents, with the third parent controlling the spread. Useful when
/// the population has started to converge.
pub fn undx_crossover(
    p1: &[f64],
    p2: &[f64],
    p3: &[f64],
    bounds: &[(f64, f64)],
    sigma_xi: f64,
    sigma_eta: f64,
    rng: &mut dyn RngCore,
) -> Vec<f64> {
    let n = p1.len();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Midpoint of p1-p2
    let midpoint: Vec<f64> = p1.iter().zip(p2).map(|(a, b)| (a + b) / 2.0).collect();
    // Direction vector p1 -> p2
    let d: Vec<f64> = p2.iter().zip(p1).map(|(b, a)| b - a).collect();
    let d_len = d.iter().map(|x| x * x).sum::<f64>().sqrt();
    if d_len < 1e-30 {
        return midpoint;
    }
    let d_unit: Vec<f64> = d.iter().map(|x| x / d_len).collect();

    // How far p3 sits from the midpoint, projected onto the orthogonal complement
    let diff: Vec<f64> = p3.iter().zip(&midpoint).map(|(a, b)| a - b).collect();
    let proj_len: f64 = diff.iter().zip(&d_unit).map(|(a, b)| a * b).sum();
    let orthogonal: Vec<f64> = diff
        .iter()
        .zip(&d_unit)
        .map(|(a, b)| a - proj_len * b)
        .collect();
    let orth_len = orthogonal.iter().map(|x| x * x).sum::<f64>().sqrt();

    let xi: f64 = normal.sample(rng) * sigma_xi;
    let child: Vec<f64> = (0..n)
        .map(|i| {
            let along = xi * d[i];
            let perturb = normal.sample(rng) * sigma_eta * orth_len;
            (midpoint[i] + along + perturb).clamp(bounds[i].0, bounds[i].1)
        })
        .collect();
    child
}
