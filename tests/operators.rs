use evo_engine::operators::binary::*;
use evo_engine::operators::crossover::*;
use evo_engine::operators::mutation::*;
use evo_engine::operators::permutation::*;
use evo_engine::operators::selection::*;
use evo_engine::Individual;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn make_rng() -> StdRng {
    StdRng::seed_from_u64(12345)
}

// ---------------------------------------------------------------------------
// Crossover tests
// ---------------------------------------------------------------------------

#[test]
fn sbx_children_stay_in_bounds() {
    let mut rng = make_rng();
    let p1 = vec![1.0, 2.0, 3.0];
    let p2 = vec![4.0, 5.0, 6.0];
    let bounds = vec![(0.0, 10.0); 3];

    for _ in 0..100 {
        let (c1, c2) = sbx_crossover(&p1, &p2, &bounds, 20.0, &mut rng);
        for i in 0..3 {
            assert!(c1[i] >= 0.0 && c1[i] <= 10.0, "c1[{}] out of bounds: {}", i, c1[i]);
            assert!(c2[i] >= 0.0 && c2[i] <= 10.0, "c2[{}] out of bounds: {}", i, c2[i]);
        }
    }
}

#[test]
fn blx_alpha_children_in_bounds() {
    let mut rng = make_rng();
    let p1 = vec![0.0, 5.0];
    let p2 = vec![10.0, 5.0];
    let bounds = vec![(0.0, 10.0); 2];

    for _ in 0..100 {
        let (c1, c2) = blx_alpha_crossover(&p1, &p2, &bounds, 0.5, &mut rng);
        for i in 0..2 {
            assert!(c1[i] >= 0.0 && c1[i] <= 10.0);
            assert!(c2[i] >= 0.0 && c2[i] <= 10.0);
        }
    }
}

#[test]
fn arithmetic_crossover_produces_intermediate_values() {
    let mut rng = make_rng();
    let p1 = vec![0.0, 0.0];
    let p2 = vec![10.0, 10.0];

    let (c1, c2) = arithmetic_crossover(&p1, &p2, &mut rng);
    for i in 0..2 {
        assert!(c1[i] >= 0.0 && c1[i] <= 10.0);
        assert!(c2[i] >= 0.0 && c2[i] <= 10.0);
    }
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

#[test]
fn polynomial_mutation_stays_in_bounds() {
    let mut rng = make_rng();
    let bounds = vec![(-5.0, 5.0); 5];

    for _ in 0..100 {
        let mut genome = vec![0.0; 5];
        polynomial_mutation(&mut genome, &bounds, 20.0, 1.0, &mut rng);
        for (i, &g) in genome.iter().enumerate() {
            assert!(g >= -5.0 && g <= 5.0, "gene {} out of bounds: {}", i, g);
        }
    }
}

#[test]
fn gaussian_mutation_stays_in_bounds() {
    let mut rng = make_rng();
    let bounds = vec![(-1.0, 1.0); 3];

    for _ in 0..200 {
        let mut genome = vec![0.0; 3];
        gaussian_mutation(&mut genome, &bounds, 10.0, 1.0, &mut rng);
        for (i, &g) in genome.iter().enumerate() {
            assert!(g >= -1.0 && g <= 1.0, "gene {} out of bounds: {}", i, g);
        }
    }
}

#[test]
fn cauchy_mutation_stays_in_bounds() {
    let mut rng = make_rng();
    let bounds = vec![(-2.0, 2.0); 4];

    for _ in 0..200 {
        let mut genome = vec![0.0; 4];
        cauchy_mutation(&mut genome, &bounds, 5.0, 1.0, &mut rng);
        for (i, &g) in genome.iter().enumerate() {
            assert!(g >= -2.0 && g <= 2.0, "gene {} out of bounds: {}", i, g);
        }
    }
}

#[test]
fn adaptive_mutator_adjusts_sigma() {
    let mut am = evo_engine::operators::mutation::AdaptiveMutator::new(1.0);
    let initial_sigma = am.sigma;

    // Report many successes, sigma should increase
    for _ in 0..25 {
        am.total_count += 1;
        am.report_success(true);
    }
    assert!(am.sigma > initial_sigma, "sigma should increase after many successes");
}

// ---------------------------------------------------------------------------
// Selection tests
// ---------------------------------------------------------------------------

#[test]
fn tournament_select_picks_best() {
    let mut rng = make_rng();
    let pop: Vec<Individual<Vec<f64>>> = (0..10)
        .map(|i| {
            let mut ind = Individual::new(vec![i as f64]);
            ind.objectives = vec![i as f64]; // fitness = index
            ind
        })
        .collect();

    // With tournament size = pop size, should always pick the best
    let selected = tournament_select(&pop, 10, &mut rng);
    assert_eq!(selected.fitness(), 0.0);
}

#[test]
fn rank_select_returns_valid_individual() {
    let mut rng = make_rng();
    let pop: Vec<Individual<Vec<f64>>> = (0..20)
        .map(|i| {
            let mut ind = Individual::new(vec![i as f64]);
            ind.objectives = vec![i as f64];
            ind
        })
        .collect();

    for _ in 0..50 {
        let selected = rank_select(&pop, &mut rng);
        assert!(selected.fitness() >= 0.0 && selected.fitness() < 20.0);
    }
}

// ---------------------------------------------------------------------------
// Binary operator tests
// ---------------------------------------------------------------------------

#[test]
fn single_point_crossover_preserves_length() {
    let mut rng = make_rng();
    let p1 = vec![true; 10];
    let p2 = vec![false; 10];

    let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
    assert_eq!(c1.len(), 10);
    assert_eq!(c2.len(), 10);
}

#[test]
fn two_point_crossover_preserves_length() {
    let mut rng = make_rng();
    let p1 = vec![true; 8];
    let p2 = vec![false; 8];

    let (c1, c2) = two_point_crossover(&p1, &p2, &mut rng);
    assert_eq!(c1.len(), 8);
    assert_eq!(c2.len(), 8);
}

#[test]
fn uniform_crossover_preserves_length() {
    let mut rng = make_rng();
    let p1 = vec![true; 6];
    let p2 = vec![false; 6];

    let (c1, c2) = uniform_crossover(&p1, &p2, &mut rng);
    assert_eq!(c1.len(), 6);
    assert_eq!(c2.len(), 6);
}

#[test]
fn bitflip_with_prob_one_flips_all() {
    let mut rng = make_rng();
    let mut genome = vec![true, false, true, false];
    bitflip_mutation(&mut genome, 1.0, &mut rng);
    assert_eq!(genome, vec![false, true, false, true]);
}

// ---------------------------------------------------------------------------
// Permutation operator tests
// ---------------------------------------------------------------------------

#[test]
fn order_crossover_produces_valid_permutation() {
    let mut rng = make_rng();
    let p1 = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let p2 = vec![7, 6, 5, 4, 3, 2, 1, 0];

    for _ in 0..50 {
        let child = order_crossover(&p1, &p2, &mut rng);
        assert_eq!(child.len(), 8);
        let mut sorted = child.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5, 6, 7], "not a valid permutation");
    }
}

#[test]
fn pmx_crossover_produces_valid_permutations() {
    let mut rng = make_rng();
    let p1 = vec![0, 1, 2, 3, 4];
    let p2 = vec![4, 3, 2, 1, 0];

    for _ in 0..50 {
        let (c1, c2) = pmx_crossover(&p1, &p2, &mut rng);
        let mut s1 = c1.clone();
        let mut s2 = c2.clone();
        s1.sort();
        s2.sort();
        assert_eq!(s1, vec![0, 1, 2, 3, 4], "c1 not a valid permutation: {:?}", c1);
        assert_eq!(s2, vec![0, 1, 2, 3, 4], "c2 not a valid permutation: {:?}", c2);
    }
}

#[test]
fn swap_mutation_preserves_permutation() {
    let mut rng = make_rng();
    let mut genome = vec![0, 1, 2, 3, 4, 5];
    for _ in 0..20 {
        swap_mutation(&mut genome, &mut rng);
        let mut sorted = genome.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
    }
}

#[test]
fn inversion_mutation_preserves_permutation() {
    let mut rng = make_rng();
    let mut genome = vec![0, 1, 2, 3, 4, 5, 6];
    for _ in 0..20 {
        inversion_mutation(&mut genome, &mut rng);
        let mut sorted = genome.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5, 6]);
    }
}
