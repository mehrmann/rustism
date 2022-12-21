pub use self::{chromosome::*, crossover::*, individual::*, mutation::*, selection::*};

use rand::RngCore;

mod chromosome;
mod crossover;
mod individual;
mod mutation;
mod selection;

pub struct GeneticAlgorithm<S> {
    selection_method: S,
    crossover_method: Box<dyn CrossoverMethod>,
    mutation_method: Box<dyn MutationMethod>,
}

impl<S> GeneticAlgorithm<S>
where
    S: SelectionMethod,
{
    pub fn new(
        selection_method: S,
        crossover_method: impl CrossoverMethod + 'static,
        mutation_method: impl MutationMethod + 'static,
    ) -> Self {
        Self {
            selection_method,
            crossover_method: Box::new(crossover_method),
            mutation_method: Box::new(mutation_method),
        }
    }

    pub fn evolve<I>(&self, rng: &mut dyn RngCore, population: &[I]) -> Vec<I>
    where
        I: Individual,
    {
        assert!(!population.is_empty());

        (0..population.len())
            .map(|_| {
                //selection
                let _parent_a = self.selection_method.select(rng, population).chromosome();
                let _parent_b = self.selection_method.select(rng, population).chromosome();

                //crossovers
                let mut child = self.crossover_method.crossover(rng, _parent_a, _parent_b);

                //mutation
                self.mutation_method.mutate(rng, &mut child);

                //create individual
                I::create(child)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use super::*;

    fn individual(genes: &[f32]) -> TestIndividual {
        let chromosome = genes.iter().cloned().collect();
        TestIndividual::create(chromosome)
    }

    #[test]
    fn test() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let ga = GeneticAlgorithm::new(
            RouletteWheelSelection::new(),
            UniformCrossover::default(),
            GaussianMutation::new(0.5, 0.5));

        let mut population = vec![
            individual(&[0.0, 0.0, 0.0]), //0.0
            individual(&[1.0, 1.0, 1.0]), //3.0
            individual(&[1.0, 2.0, 1.0]), //4.0
            individual(&[1.0, 2.0, 4.0]), //7.0
        ];
        let initial_fitness : f32 = population.iter().map(|i| i.fitness()).sum();
        assert_eq!(initial_fitness, 14.0);

        for _ in 0..10 {
            population = ga.evolve(&mut rng, &population);
        }

        let expected_population = vec![
            individual(&[0.4476949, 2.0648358, 4.3058133]),
            individual(&[1.2126867, 1.5538777, 2.886911]),
            individual(&[1.0617678, 2.265739, 4.428764]),
            individual(&[0.95909685, 2.4618788, 4.024733]),
        ];
        assert_eq!(population, expected_population);

        let final_fitness : f32 = population.iter().map(|i| i.fitness()).sum();
        assert!(final_fitness > initial_fitness);
    }
}
