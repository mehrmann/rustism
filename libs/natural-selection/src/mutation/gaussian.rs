use crate::*;
use rand::{Rng, RngCore};

#[derive(Clone, Debug)]
pub struct GaussianMutation {
    //the chance of mutation 0.0..1.0
    chance: f32,
    //the maximum amount of mutation
    coefficient: f32,
}

impl GaussianMutation {
    pub fn new(chance: f32, coefficient: f32) -> Self {
        assert!(chance >= 0.0 && chance <= 1.0);
        Self { chance, coefficient }
    }
}

impl MutationMethod for GaussianMutation {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome) {
        for gene in child.iter_mut() {
            let sign = if rng.gen_bool(0.5) { -1.0 } else { 1.0 };
            if rng.gen_bool(self.chance as _) {
                *gene += sign * self.coefficient * rng.gen::<f32>();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn mutate(chance: f32, coefficient: f32) -> Vec<f32> {
        let mut child = vec![1.0, 2.0, 3.0, 4.0, 5.0].into_iter().collect();
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        GaussianMutation::new(chance, coefficient).mutate(&mut rng, &mut child);

        child.into_iter().collect()
    }
    mod given_zero_chance {

        fn mutate(coefficient: f32) -> Vec<f32> {
            super::mutate(0.0, coefficient)
        }

        mod and_zero_coefficient {
            use super::*;

            #[test]
            fn chromosome_should_not_be_modified() {
                let child = mutate(0.0);
                let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];
                approx::assert_relative_eq!(child.as_slice(), expected.as_slice());
            }
        }
        mod and_nonzero_coefficient {
            use super::*;

            #[test]
            fn chromosome_should_not_be_modified() {
                let child = mutate(1.0);
                let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];
                approx::assert_relative_eq!(child.as_slice(), expected.as_slice());
            }
        }
    }
    mod given_fifty_percent_chance {

        fn mutate(coefficient: f32) -> Vec<f32> {
            super::mutate(0.5, coefficient)
        }

        mod and_zero_coefficient {
            use super::*;

            #[test]
            fn chromosome_should_not_be_modified() {
                let child = mutate(0.0);
                let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];
                approx::assert_relative_eq!(child.as_slice(), expected.as_slice());
            }
        }
        mod and_nonzero_coefficient {
            use super::*;

            #[test]
            fn chromosome_should_be_slightly_modified() {
                let child = mutate(0.5);
                let expected = vec![1.0, 1.7756249, 3.0, 4.1596804, 5.0];
                approx::assert_relative_eq!(child.as_slice(), expected.as_slice());
            }
        }
    }
    mod given_max_chance {

        fn mutate(coefficient: f32) -> Vec<f32> {
            super::mutate(1.0, coefficient)
        }

        mod and_zero_coefficient {
            use super::*;

            #[test]
            fn chromosome_should_not_be_modified() {
                let child = mutate(0.0);
                let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];
                approx::assert_relative_eq!(child.as_slice(), expected.as_slice());
            }
        }
        mod and_nonzero_coefficient {
            use super::*;

            #[test]
            fn chromosome_is_totally_modified() {
                let child = mutate(0.5);
                let expected = vec![1.4545316, 2.1162078, 2.7756248, 3.9505124, 4.638691];
                approx::assert_relative_eq!(child.as_slice(), expected.as_slice());
            }
        }
    }
}
