use std::ops::{Add, Index};

#[derive(Clone, Debug)]
pub struct Chromosome {
    genes: Vec<f32>,
}

impl Chromosome {
    pub fn len(&self) -> usize {
        self.genes.len()
    }

    pub fn dosomething() -> Vec<f32> {
        todo!()
    }

    pub fn iter(&self) -> impl Iterator<Item=&f32> {
        self.genes.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item=&mut f32> {
        self.genes.iter_mut()
    }

    fn value_to_string(mut x: u32) -> String {
        let mut result = vec![];

        loop {
            let m = x % 52;
            x = x / 52;

            if m<26 {
                result.push((m as u8 + b'a') as char);
            } else {
                result.push((m as u8 - 26 + b'A' ) as char);
            }
            if x == 0 {
                break;
            }
        }
        result.into_iter().rev().collect()
    }

    fn string_to_value(s: &str) -> u32 {
        let mut v : u32 = 0;
        for c in s.chars() {
            v*=52;
            if c>='A' && c<='Z' {
                v += 26 + (c as u8 - b'A') as u32;
            } else if c>='a' && c<='z' {
                v += (c as u8 - b'a') as u32;
            } else {
                panic!("unexpected character encountered!")
            }
        }
        v
    }

    pub fn to_dna(&self) -> String {
        let mut result = self.genes.iter().fold(String::new(), |mut result, &f| {
            let f = ((f + 1000.0) * 1000.0) as u32;
            result = result.add(&Self::value_to_string(f));
            result.push('-');
            result
        });
        result.pop(); // remove the last dash
        result
    }

    pub fn from_dna(dna: String) -> Self {
        let genes = dna.split('-').map(|s| (Self::string_to_value(s) as f32 / 1000.0) - 1000.0).collect();
        Self { genes }
    }
}

impl Index<usize> for Chromosome {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.genes[index]
    }
}

impl FromIterator<f32> for Chromosome {
    fn from_iter<T: IntoIterator<Item=f32>>(iter: T) -> Self {
        Self {
            genes: iter.into_iter().collect(),
        }
    }
}

impl IntoIterator for Chromosome {
    type Item = f32;
    /* with
       #![feature(min_type_alias_impl_trait)]
       this could be
       type IntoIter = impl Iterator<Item = f32>;
    */
    type IntoIter = std::vec::IntoIter<f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.genes.into_iter()
    }
}

#[cfg(test)]
impl PartialEq for Chromosome {
    fn eq(&self, other: &Self) -> bool {
        approx::relative_eq!(self.genes.as_slice(), other.genes.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use crate::chromosome::Chromosome;

    fn chromosome() -> Chromosome {
        Chromosome {
            genes: vec![3.0, 1.0, 2.0],
        }
    }

    mod len {
        use super::*;

        #[test]
        fn test() {
            assert_eq!(chromosome().len(), 3);
        }
    }

    mod index {
        use super::*;

        #[test]
        fn test() {
            let chromosome = chromosome();

            assert_eq!(chromosome[0], 3.0);
            assert_eq!(chromosome[1], 1.0);
            assert_eq!(chromosome[2], 2.0);
        }
    }

    mod from_iterator {
        use super::*;

        #[test]
        fn test() {
            let chromosome: Chromosome = vec![3.0, 1.0, 2.0].into_iter().collect();

            assert_eq!(chromosome[0], 3.0);
            assert_eq!(chromosome[1], 1.0);
            assert_eq!(chromosome[2], 2.0);
        }
    }

    mod into_iterator {
        use super::*;

        #[test]
        fn test() {
            let chromosome = Chromosome {
                genes: vec![3.0, 1.0, 2.0],
            };

            let genes: Vec<_> = chromosome.into_iter().collect();

            assert_eq!(genes.len(), 3);
            assert_eq!(genes[0], 3.0);
            assert_eq!(genes[1], 1.0);
            assert_eq!(genes[2], 2.0);
        }
    }

    mod iter {
        use super::*;

        #[test]
        fn test() {
            let chromosome = chromosome();
            let genes: Vec<_> = chromosome.iter().collect();

            assert_eq!(genes.len(), 3);
            assert_eq!(genes[0], &3.0);
            assert_eq!(genes[1], &1.0);
            assert_eq!(genes[2], &2.0);
        }
    }

    mod iter_mut {
        use super::*;

        #[test]
        fn test() {
            let mut chromosome = chromosome();

            chromosome.iter_mut().for_each(|gene| *gene *= 10.0);

            let genes: Vec<_> = chromosome.iter().collect();

            assert_eq!(genes.len(), 3);
            assert_eq!(genes[0], &30.0);
            assert_eq!(genes[1], &10.0);
            assert_eq!(genes[2], &20.0);
        }
    }

    mod dna {
        use super::*;

        mod to_dna {
            use super::*;

            #[test]
            fn test() {
                let dna = Chromosome {
                    genes: vec![-100.0, -50.0, 0.0, 1.0, 2.0, 3.0, 50.0, 100.0],
                }.to_dna();
                assert_eq!(dna, "guRK-gNrm-hfQO-hgka-hgDm-hgWy-hyqq-hQPS");
            }
        }

        mod conversion {

            use super::*;

            #[test]
            fn test_value_to_string() {
                assert_eq!(Chromosome::value_to_string(0), "a");
                assert_eq!(Chromosome::value_to_string(1000), "tm");
                assert_eq!(Chromosome::value_to_string(2000), "My");
                assert_eq!(Chromosome::value_to_string(((0.5 + 1000.0) * 1000.0) as u32), "hgau");
            }

            #[test]
            fn test_string_to_value() {
                assert_eq!(Chromosome::string_to_value("a"), 0);
                assert_eq!(Chromosome::string_to_value("tm"), 1000);
                assert_eq!(Chromosome::string_to_value("My"), 2000);
                assert_eq!(Chromosome::string_to_value("hgau") as f64 / 1000.0 - 1000.0, 0.5);
            }
        }

        mod from_dna {
            use super::*;

            #[test]
            fn test() {
                let actual = Chromosome::from_dna("guRK-gNrm-hfQO-hgka-hgDm-hgWy-hyqq-hQPS".to_string());
                let dna = Chromosome {
                    genes: vec![-100.0, -50.0, 0.0, 1.0, 2.0, 3.0, 50.0, 100.0],};
                assert_eq!(actual, dna);
            }
        }
    }
}
