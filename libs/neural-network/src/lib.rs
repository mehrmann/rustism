extern crate core;

use std::iter::once;
use rand::prelude::*;

pub struct Network {
    layers: Vec<Layer>,
}

struct Layer {
    neurons: Vec<Neuron>,
}

#[derive(Debug, Clone, PartialEq)]
struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

pub struct LayerTopology {
    pub neurons: usize,
}

impl Network {
    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(inputs))
    }

    pub fn random(rng: &mut dyn RngCore, layers: &[LayerTopology]) -> Self {
        assert!(layers.len() > 1); //needs to have more than 1 layer

        let layers = layers
            .windows(2)
            .map(|layers| Layer::random(rng, layers[0].neurons, layers[1].neurons))
            .collect();

        Self { layers }
    }

    pub fn from_data(layers: &[LayerTopology], data: impl IntoIterator<Item = f32>) -> Self {
        assert!(layers.len() > 1); //needs to have more than 1 layer

        let mut data = data.into_iter();

        let layers = layers
            .windows(2)
            .map(|layers| Layer::from_data(layers[0].neurons, layers[1].neurons, &mut data))
            .collect();

        if data.next().is_some() {
            panic!("something is wrong...");
        }

        Self { layers }
    }

    pub fn data(&self) -> impl Iterator<Item = f32> + '_ {
        self.layers.iter()
            .flat_map(|layer| layer.neurons.iter())
            .flat_map(|neuron| once(&neuron.bias).chain(&neuron.weights))
            .cloned()
    }


}

impl Layer {
    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        assert_eq!(inputs.len(), self.neurons[0].weights.len());

        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(&inputs))
            .collect()
    }

    fn random(rng: &mut dyn RngCore, input_neurons: usize, output_neurons: usize) -> Self {
        let neurons = (0..output_neurons)
            .map(|_| Neuron::random(rng, input_neurons))
            .collect();

        Self { neurons }
    }

    fn from_data(input_neurons: usize, output_neurons: usize, data: &mut dyn Iterator<Item = f32>) -> Self {
        let neurons = (0..output_neurons)
            .map(|_| Neuron::from_data(input_neurons, data))
            .collect();

        Self { neurons }

    }

}

impl Neuron {
    fn propagate(&self, inputs: &[f32]) -> f32 {
        assert_eq!(inputs.len(), self.weights.len());

        let output = inputs
            .iter()
            .zip(&self.weights) //zip inputs and weights
            .map(|(input, weight)| input * weight) //calculate weighted inputs
            .sum::<f32>(); //sum up weighted inputs

        (self.bias + output).max(0.0) //ReLu
    }

    fn random(rng: &mut dyn RngCore, output_size: usize) -> Self {
        let bias = rng.gen_range(-1.0..=1.0);
        let weights = (0..output_size)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();

        Self { bias, weights }
    }

    fn from_data(output_size: usize, data: &mut dyn Iterator<Item = f32>) -> Self {
        let bias = data.next().expect("out of data");
        let weights = (0..output_size)
            .map(|_| data.next().expect("out of data"))
            .collect();

        Self { bias, weights }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand_chacha::ChaCha8Rng;

    mod random {
        use super::*;

        #[test]
        fn test_neuron() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let neuron_a = Neuron::random(&mut rng, 4);

            assert_relative_eq!(neuron_a.bias, -0.6255188);
            assert_relative_eq!(
                neuron_a.weights.as_slice(),
                [0.67383957, 0.8181262, 0.26284897, 0.5238807].as_ref()
            );
        }
    }

    mod propagate {
        use super::*;

        #[test]
        fn test_neuron() {
            let neuron = Neuron {
                bias: 0.5,
                weights: vec![-0.3, 0.8],
            };

            assert_relative_eq!(neuron.propagate(&[-10.0, -10.0]), 0.0,);

            assert_relative_eq!(
                neuron.propagate(&[0.5, 1.0]),
                (-0.3 * 0.5) + (0.8 * 1.0) + 0.5,
            );
        }

        #[test]
        fn test_layer_propagate() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());

            let layer = Layer::random(&mut rng, 5, 5);
            let results = layer.propagate((0..5).map(|_| rng.gen_range(-1.0..=1.0)).collect());

            assert_relative_eq!(
                results.as_slice(),
                [0.93348336, 0.0, 0.0, 0.3035017, 0.0].as_ref()
            );
        }

        #[test]
        fn test_network_propagate() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());

            let network = Network::random(
                &mut rng,
                &[
                    LayerTopology { neurons: 8 },
                    LayerTopology { neurons: 4 },
                    LayerTopology { neurons: 3 },
                ],
            );

            let results = network.propagate((0..8).map(|_| rng.gen_range(0.0..=1.0)).collect());
            assert_relative_eq!(results.as_slice(), [1.6144389, 0.0, 1.0972998].as_ref());
        }

        #[test]
        fn test_dna_restore() {
            let topology = &[
                LayerTopology { neurons: 3 },
                LayerTopology { neurons: 2 },
            ];
            let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7];

            let actual : Vec<_> = Network::from_data(
                topology,
                weights.clone()
            ).data().collect();

            assert_relative_eq!(actual.as_slice(), weights.as_slice());
        }
    }
}
