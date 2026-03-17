//! Combinatorial benchmark problems using permutation and binary genomes.

use crate::{Individual, Problem};
use rand::seq::SliceRandom;
use rand::Rng;

// ---------------------------------------------------------------------------
// OneMax (binary): maximize the number of 1s in a bitstring.
// We frame it as minimization: fitness = n - count_ones.
// ---------------------------------------------------------------------------

pub struct OneMax {
    pub length: usize,
}

impl Problem for OneMax {
    type Genome = Vec<bool>;

    fn num_objectives(&self) -> usize {
        1
    }
    fn dimension(&self) -> usize {
        self.length
    }
    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(0.0, 1.0); self.length]
    }

    fn evaluate(&self, ind: &mut Individual<Vec<bool>>) {
        let ones = ind.genome.iter().filter(|&&b| b).count();
        ind.objectives = vec![(self.length - ones) as f64];
    }

    fn random_genome(&self, rng: &mut impl Rng) -> Vec<bool> {
        (0..self.length).map(|_| rng.gen_bool(0.5)).collect()
    }
}

// ---------------------------------------------------------------------------
// TSP (permutation): Traveling Salesman Problem.
// Given a distance matrix, find the shortest tour visiting every city once.
// ---------------------------------------------------------------------------

pub struct Tsp {
    /// Symmetric distance matrix. distances[i][j] = cost from city i to city j.
    pub distances: Vec<Vec<f64>>,
}

impl Tsp {
    /// Create a TSP instance from a list of 2D city coordinates.
    /// Distances are Euclidean.
    pub fn from_coordinates(coords: &[(f64, f64)]) -> Self {
        let n = coords.len();
        let mut distances = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = coords[i].0 - coords[j].0;
                let dy = coords[i].1 - coords[j].1;
                let d = (dx * dx + dy * dy).sqrt();
                distances[i][j] = d;
                distances[j][i] = d;
            }
        }
        Self { distances }
    }
}

impl Problem for Tsp {
    type Genome = Vec<usize>;

    fn num_objectives(&self) -> usize {
        1
    }
    fn dimension(&self) -> usize {
        self.distances.len()
    }
    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(0.0, self.distances.len() as f64); self.distances.len()]
    }

    fn evaluate(&self, ind: &mut Individual<Vec<usize>>) {
        let n = ind.genome.len();
        let mut total = 0.0;
        for i in 0..n {
            let from = ind.genome[i];
            let to = ind.genome[(i + 1) % n];
            total += self.distances[from][to];
        }
        ind.objectives = vec![total];
    }

    fn random_genome(&self, rng: &mut impl Rng) -> Vec<usize> {
        let mut perm: Vec<usize> = (0..self.distances.len()).collect();
        perm.shuffle(rng);
        perm
    }
}

// ---------------------------------------------------------------------------
// Knapsack (binary): maximize value without exceeding weight capacity.
// Framed as minimization: fitness = -value + penalty * max(0, weight - capacity).
// ---------------------------------------------------------------------------

pub struct Knapsack {
    pub weights: Vec<f64>,
    pub values: Vec<f64>,
    pub capacity: f64,
    /// Penalty multiplier for exceeding capacity.
    pub penalty: f64,
}

impl Problem for Knapsack {
    type Genome = Vec<bool>;

    fn num_objectives(&self) -> usize {
        1
    }
    fn dimension(&self) -> usize {
        self.weights.len()
    }
    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(0.0, 1.0); self.weights.len()]
    }

    fn evaluate(&self, ind: &mut Individual<Vec<bool>>) {
        let total_weight: f64 = ind
            .genome
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| self.weights[i])
            .sum();
        let total_value: f64 = ind
            .genome
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| self.values[i])
            .sum();

        let excess = (total_weight - self.capacity).max(0.0);
        ind.constraint_violation = excess;
        // Negate value since we minimize; add penalty for overweight
        ind.objectives = vec![-total_value + self.penalty * excess];
    }

    fn random_genome(&self, rng: &mut impl Rng) -> Vec<bool> {
        (0..self.weights.len()).map(|_| rng.gen_bool(0.5)).collect()
    }
}
