/// Solves a 0/1 knapsack problem using binary operators and a simple GA loop.
///
/// This shows how to use the binary genome operators with the Knapsack
/// problem definition from evo_engine::problems::combinatorial.
use evo_engine::operators::binary::{bitflip_mutation, single_point_crossover};
use evo_engine::problems::combinatorial::Knapsack;
use evo_engine::{Individual, Problem};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() {
    // 20 items with random weights and values
    let knapsack = Knapsack {
        weights: vec![
            12.0, 7.0, 11.0, 8.0, 9.0, 14.0, 3.0, 6.0, 10.0, 5.0, 13.0, 4.0, 15.0, 2.0, 8.0, 11.0,
            7.0, 6.0, 9.0, 3.0,
        ],
        values: vec![
            24.0, 13.0, 23.0, 15.0, 16.0, 28.0, 5.0, 10.0, 21.0, 9.0, 25.0, 7.0, 29.0, 3.0, 14.0,
            20.0, 12.0, 11.0, 17.0, 4.0,
        ],
        capacity: 50.0,
        penalty: 100.0,
    };

    let mut rng = StdRng::seed_from_u64(99);
    let pop_size = 80;
    let generations = 300;

    let mut pop: Vec<Individual<Vec<bool>>> = (0..pop_size)
        .map(|_| {
            let genome = knapsack.random_genome(&mut rng);
            let mut ind = Individual::new(genome);
            knapsack.evaluate(&mut ind);
            ind
        })
        .collect();

    for gen in 0..generations {
        // Sort by constrained comparison (feasible solutions first)
        pop.sort_by(|a, b| a.constrained_cmp(b));

        if gen % 50 == 0 {
            let best = &pop[0];
            let value: f64 = best
                .genome
                .iter()
                .enumerate()
                .filter(|(_, &b)| b)
                .map(|(i, _)| knapsack.values[i])
                .sum();
            let weight: f64 = best
                .genome
                .iter()
                .enumerate()
                .filter(|(_, &b)| b)
                .map(|(i, _)| knapsack.weights[i])
                .sum();
            println!(
                "Gen {:4} | value: {:.0}, weight: {:.0}/{:.0}, feasible: {}",
                gen,
                value,
                weight,
                knapsack.capacity,
                best.is_feasible()
            );
        }

        let mut next_pop: Vec<Individual<Vec<bool>>> = pop[..2].to_vec();

        while next_pop.len() < pop_size {
            // Tournament selection
            let pick = |rng: &mut StdRng| -> usize {
                let a = rng.gen_range(0..pop.len());
                let b = rng.gen_range(0..pop.len());
                if pop[a].constrained_cmp(&pop[b]) == std::cmp::Ordering::Less {
                    a
                } else {
                    b
                }
            };

            let p1 = pick(&mut rng);
            let p2 = pick(&mut rng);

            let (mut c1g, mut c2g) =
                single_point_crossover(&pop[p1].genome, &pop[p2].genome, &mut rng);

            bitflip_mutation(&mut c1g, 1.0 / 20.0, &mut rng);
            bitflip_mutation(&mut c2g, 1.0 / 20.0, &mut rng);

            let mut c1 = Individual::new(c1g);
            knapsack.evaluate(&mut c1);
            next_pop.push(c1);

            if next_pop.len() < pop_size {
                let mut c2 = Individual::new(c2g);
                knapsack.evaluate(&mut c2);
                next_pop.push(c2);
            }
        }

        pop = next_pop;
    }

    pop.sort_by(|a, b| a.constrained_cmp(b));
    let best = &pop[0];
    let items: Vec<usize> = best
        .genome
        .iter()
        .enumerate()
        .filter(|(_, &b)| b)
        .map(|(i, _)| i)
        .collect();
    let value: f64 = items.iter().map(|&i| knapsack.values[i]).sum();
    let weight: f64 = items.iter().map(|&i| knapsack.weights[i]).sum();

    println!("\nBest solution:");
    println!("  Items: {:?}", items);
    println!("  Value: {:.0}", value);
    println!("  Weight: {:.0} / {:.0}", weight, knapsack.capacity);
}
