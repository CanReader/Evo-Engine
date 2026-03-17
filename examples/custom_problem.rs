/// Shows how to define your own optimization problem and solve it with GA.
///
/// We'll minimize f(x, y) = (x - 3)^2 + (y + 1)^2
/// The answer is obviously (3, -1) with f = 0, but let's pretend
/// we don't know that.
use evo_engine::algorithms::ga::GeneticAlgorithm;
use evo_engine::{EvolutionConfig, EvolutionaryAlgorithm, Individual, Problem};
use rand::Rng;

struct MyProblem;

impl Problem for MyProblem {
    type Genome = Vec<f64>;

    fn num_objectives(&self) -> usize {
        1
    }
    fn dimension(&self) -> usize {
        2
    }
    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(-10.0, 10.0), (-10.0, 10.0)]
    }
    fn random_genome(&self, rng: &mut impl Rng) -> Vec<f64> {
        self.bounds()
            .iter()
            .map(|(lo, hi)| rng.gen::<f64>() * (hi - lo) + lo)
            .collect()
    }
    fn evaluate(&self, ind: &mut Individual<Vec<f64>>) {
        let x = ind.genome[0];
        let y = ind.genome[1];
        ind.objectives = vec![(x - 3.0).powi(2) + (y + 1.0).powi(2)];
    }
}

fn main() {
    let ga = GeneticAlgorithm::default();
    let config = EvolutionConfig {
        population_size: 100,
        max_generations: 200,
        target_fitness: Some(1e-8),
        elitism_count: 2,
        seed: Some(42),
    };

    let result = ga.run(&MyProblem, &config).expect("optimization failed");

    println!("Best fitness: {:.6e}", result.best.fitness());
    println!(
        "Best solution: x = {:.4}, y = {:.4}",
        result.best.genome[0], result.best.genome[1]
    );
    println!("Generations: {}", result.generations_run);
}
