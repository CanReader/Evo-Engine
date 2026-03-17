/// Solves a small Traveling Salesman Problem using a simple
/// permutation-based GA built on evo-engine's operators.
///
/// This example wires up the permutation operators manually since
/// the built-in GA is designed for real-valued genomes. It shows
/// how to use the lower-level operator functions directly.
use evo_engine::operators::permutation::{order_crossover, inversion_mutation};
use evo_engine::problems::combinatorial::Tsp;
use evo_engine::{Individual, Problem};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() {
    // 15 random cities in a 100x100 grid
    let coords: Vec<(f64, f64)> = vec![
        (20.0, 30.0), (50.0, 90.0), (10.0, 10.0), (80.0, 60.0),
        (40.0, 50.0), (70.0, 20.0), (60.0, 80.0), (30.0, 70.0),
        (90.0, 40.0), (15.0, 55.0), (55.0, 35.0), (75.0, 85.0),
        (25.0, 15.0), (85.0, 75.0), (45.0, 65.0),
    ];
    let tsp = Tsp::from_coordinates(&coords);
    let mut rng = StdRng::seed_from_u64(42);
    let pop_size = 100;
    let generations = 500;
    let mutation_rate = 0.3;

    // Create initial population
    let mut pop: Vec<Individual<Vec<usize>>> = (0..pop_size)
        .map(|_| {
            let genome = tsp.random_genome(&mut rng);
            let mut ind = Individual::new(genome);
            tsp.evaluate(&mut ind);
            ind
        })
        .collect();

    for gen in 0..generations {
        pop.sort_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap());

        if gen % 100 == 0 {
            println!(
                "Gen {:4} | best tour length: {:.2}",
                gen,
                pop[0].fitness()
            );
        }

        // Elitism: keep the top 2
        let mut next_pop: Vec<Individual<Vec<usize>>> = pop[..2].to_vec();

        while next_pop.len() < pop_size {
            // Tournament selection (k=3)
            let pick = |rng: &mut StdRng| -> &Individual<Vec<usize>> {
                let mut best = rng.gen_range(0..pop.len());
                for _ in 0..2 {
                    let idx = rng.gen_range(0..pop.len());
                    if pop[idx].fitness() < pop[best].fitness() {
                        best = idx;
                    }
                }
                &pop[best]
            };

            let p1 = pick(&mut rng);
            let p2 = pick(&mut rng);

            let mut child_genome = order_crossover(&p1.genome, &p2.genome, &mut rng);

            if rng.gen::<f64>() < mutation_rate {
                inversion_mutation(&mut child_genome, &mut rng);
            }

            let mut child = Individual::new(child_genome);
            tsp.evaluate(&mut child);
            next_pop.push(child);
        }

        pop = next_pop;
    }

    pop.sort_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap());
    println!("\nFinal best tour length: {:.2}", pop[0].fitness());
    println!("Tour: {:?}", pop[0].genome);
}
