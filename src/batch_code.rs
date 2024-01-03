use std::collections::{HashMap, HashSet};

use aes_prng::AesRng;
use rand::{CryptoRng, RngCore, SeedableRng};

use crate::{dpf::BitSlice, BitVec, Dmpf, DmpfKey, DpfKey, Node};

pub struct BatchCodeDmpf {
    input_domain_log_size: usize,
    hash_functions_count: usize,
    expansion_overhead_in_percent: usize,
}

pub struct BatchCodeDmpfKey {
    input_domain_log_size: usize,
    log_bucket_size: usize,
    buckets: Vec<DpfKey>,
    hash_functions: Vec<HashFunction>,
}

impl Dmpf for BatchCodeDmpf {
    type Key = BatchCodeDmpfKey;
    fn try_gen<R: rand::prelude::CryptoRng + rand::prelude::RngCore>(
        &self,
        inputs: &[(
            <Self::Key as DmpfKey>::InputContainer,
            <Self::Key as DmpfKey>::OutputContainer,
        )],
        mut rng: &mut R,
    ) -> Option<(Self::Key, Self::Key)> {
        let buckets = (inputs.len() * (100 + self.expansion_overhead_in_percent)) / 100;
        let bucket_size = ((1 << self.input_domain_log_size) * self.hash_functions_count) / buckets;
        let bucket_log_size = usize::ilog2(bucket_size) as usize;
        assert_eq!(1 << bucket_log_size, bucket_size);
        let index_to_value_map: HashMap<_, _> = inputs
            .iter()
            .map(|v| {
                (
                    (v.0 >> ((u128::BITS as usize) - self.input_domain_log_size)) as usize,
                    v.1,
                )
            })
            .collect();
        let indices: Vec<_> = index_to_value_map.keys().copied().collect();
        let hash_functions_seeds: Vec<_> = (0..self.hash_functions_count)
            .map(|_| {
                let mut seed = [0; aes_prng::SEED_SIZE];
                rng.fill_bytes(&mut seed);
                seed
            })
            .collect();
        let encoding = batch_encode::<AesRng>(
            self.input_domain_log_size,
            &indices[..],
            buckets,
            &hash_functions_seeds[..],
            &mut rng,
        );
        let mut dpfs: Vec<_> = (0..buckets)
            .map(|_| {
                let roots = (Node::random(&mut rng), Node::random(&mut rng));
                let alpha = BitVec::new(bucket_log_size);
                let beta = BitVec::new(128);
                DpfKey::gen(&roots, &alpha, &beta)
            })
            .collect();
        for (index, (bucket, index_in_bucket)) in encoding {
            let value = index_to_value_map[&index];
            let mut alpha = BitVec::new(bucket_log_size);
            alpha.as_mut()[0] =
                ((index_in_bucket as u128) << (128 - self.input_domain_log_size)).into();
            let mut beta = BitVec::new(128);
            beta.as_mut()[0] = value.into();
            let roots = (Node::random(&mut rng), Node::random(&mut rng));
            dpfs[bucket] = DpfKey::gen(&roots, &alpha, &beta);
        }
        let (dpfs_0, dpfs_1): (Vec<_>, Vec<_>) = dpfs.into_iter().unzip();
        let hash_functions = gen_hash_functions::<AesRng>(
            self.input_domain_log_size,
            bucket_size,
            &hash_functions_seeds[..],
        );
        Some((
            BatchCodeDmpfKey {
                input_domain_log_size: self.input_domain_log_size,
                log_bucket_size: bucket_log_size,
                buckets: dpfs_0,
                hash_functions: hash_functions.clone(),
            },
            BatchCodeDmpfKey {
                input_domain_log_size: self.input_domain_log_size,
                log_bucket_size: bucket_log_size,
                buckets: dpfs_1,
                hash_functions,
            },
        ))
    }
}

impl DmpfKey for BatchCodeDmpfKey {
    type InputContainer = u128;
    type OutputContainer = u128;
    type Session = ();
    fn eval(&self, input: &Self::InputContainer, output: &mut Self::OutputContainer) {
        let input_usize = (input >> (128 - self.input_domain_log_size)) as usize;
        let mut output = [Node::default()];
        for f in self.hash_functions.iter() {
            let (bucket, index) = f.eval(input_usize);
            let node = [Node::from((index as u128) << (128 - self.log_bucket_size))];
            let input_bitvec = BitSlice::new(self.log_bucket_size, &node[..]);
            self.buckets[bucket].eval(&input_bitvec, output)
        }
    }
    fn eval_all(&self) -> Box<[Self::OutputContainer]> {
        unimplemented!()
    }
    fn make_session(&self) -> Self {
        unimplemented!()
    }
}

#[derive(Clone)]
struct HashFunction {
    permutation: Vec<usize>,
    bucket_size: usize,
    base_bucket: usize,
}
impl HashFunction {
    fn new<R: RngCore + CryptoRng + SeedableRng>(
        input_domain_log_size: usize,
        bucket_size: usize,
        mut rng: R,
        bucket_id: usize,
    ) -> Self {
        let input_domain_size: usize = 1 << input_domain_log_size;
        let permutation = Self::random_permutation(input_domain_size, rng);
        let base_bucket = input_domain_size.div_ceil(bucket_size) * bucket_id;
        Self {
            permutation,
            bucket_size,
            base_bucket,
        }
    }
    fn eval(&self, item: usize) -> (usize, usize) {
        let v = self.permutation[item];
        (
            self.base_bucket + (v / self.bucket_size),
            v % self.bucket_size,
        )
    }
    fn random_permutation<R: CryptoRng + RngCore>(n: usize, mut rng: R) -> Vec<usize> {
        let mut output: Vec<_> = (0..n).collect();
        for i in 1..n {
            let r = (rng.next_u64() as usize) % (i + 1);
            output.swap(i, r);
        }
        output
    }
}
fn gen_hash_functions<R: RngCore + CryptoRng + SeedableRng>(
    input_domain_log_size: usize,
    bucket_size: usize,
    seeds: &[R::Seed],
) -> Vec<HashFunction>
where
    R::Seed: Clone + Copy,
{
    seeds
        .iter()
        .enumerate()
        .map(|(idx, s)| {
            let mut rng = R::from_seed(*s);
            HashFunction::new(input_domain_log_size, bucket_size, rng, idx)
        })
        .collect()
}

fn batch_encode<R: CryptoRng + RngCore + SeedableRng>(
    input_domain_log_size: usize,
    indices: &[usize],
    buckets: usize,
    hash_functions_seeds: &[R::Seed],
    mut rng: impl CryptoRng + RngCore,
) -> Vec<(usize, (usize, usize))>
where
    R::Seed: Clone + Copy,
{
    let input_domain_size = 1 << input_domain_log_size;
    let hash_functions_num = hash_functions_seeds.len();
    let bucket_size = (input_domain_size * hash_functions_num) / buckets;
    assert_eq!(
        bucket_size * buckets,
        input_domain_size * hash_functions_num
    );
    let hash_functions =
        gen_hash_functions::<R>(input_domain_log_size, bucket_size, hash_functions_seeds);
    let mut item_to_bucket = vec![None; indices.len()];
    let mut bucket_to_item = vec![None; buckets];
    let mut bucket_randomness = rng.next_u64() as usize;
    let mut indices_left = HashSet::<_>::from_iter(0..indices.len());
    let mut index_to_map = (rng.next_u64() as usize) % indices.len();
    assert!(indices_left.remove(&index_to_map));
    let mut iterations_without_advance = 0;
    for _ in 0..(indices.len() * indices.len()) {
        let item_to_map = indices[index_to_map];
        // Sample random function
        let function_id = bucket_randomness % hash_functions_num;
        bucket_randomness /= hash_functions_num;
        if bucket_randomness < hash_functions_num {
            bucket_randomness = rng.next_u64() as usize;
        }

        let (bucket, index_in_bucket) = hash_functions[function_id].eval(indices[item_to_map]);

        index_to_map = if let Some(v) = bucket_to_item[bucket].replace(index_to_map) {
            item_to_bucket[index_to_map] = Some((bucket, index_in_bucket));
            item_to_bucket[v] = None;
            iterations_without_advance += 1;
            if iterations_without_advance <= indices.len() * hash_functions_num {
                v
            } else {
                panic!("Can't find matching");
            }
        } else {
            item_to_bucket[index_to_map] = Some((bucket, index_in_bucket));
            iterations_without_advance = 0;
            let v = match indices_left.iter().next() {
                None => break,
                Some(v) => *v,
            };
            indices_left.remove(&v);
            v
        }
    }
    item_to_bucket
        .into_iter()
        .enumerate()
        .map(|(idx, v)| (indices[idx], v.unwrap()))
        .collect()
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use aes_prng::AesRng;
    use rand::thread_rng;

    use crate::batch_code::gen_hash_functions;

    use super::batch_encode;

    #[test]
    fn test_batch_code() {
        let input_domain_log_size: usize = 15;
        let input_domain_size = 1 << input_domain_log_size;
        let indices_count: usize = 1 << 10;
        let hash_functions_count = 3;
        let indices = make_indices(indices_count);
        let seeds = make_seeds(hash_functions_count);
        let buckets = indices_count * 3 / 2;
        let bucket_size = (input_domain_size * seeds.len()) / buckets;
        let encoding = batch_encode::<AesRng>(
            input_domain_log_size,
            &indices[..],
            buckets,
            &seeds[..],
            thread_rng(),
        );

        // Ensure we receive a mapping for each input.
        assert_eq!(encoding.len(), indices.len());

        let hash_functions =
            gen_hash_functions::<AesRng>(input_domain_log_size, bucket_size, &seeds[..]);

        // Assert all buckets a output of some hash function for the given index.
        for i in indices.iter().copied() {
            let (bucket, index_in_bucket) = encoding
                .iter()
                .copied()
                .find_map(|v| if v.0 == i { Some(v.1) } else { None })
                .unwrap();
            assert!(hash_functions.iter().any(|f| f.eval(i).0 == bucket));
            assert!(index_in_bucket < bucket_size);
        }

        // Asserts all buckets are distinct.
        let set: HashSet<_> = encoding.iter().map(|(_, b)| *b).collect();
        assert_eq!(set.len(), indices.len());
    }
    fn make_seed(i: usize) -> [u8; aes_prng::SEED_SIZE] {
        (i as u128).to_be_bytes()
    }
    fn make_seeds(count: usize) -> Vec<[u8; aes_prng::SEED_SIZE]> {
        (0..count).map(|i| make_seed(i)).collect()
    }
    fn make_indices(count: usize) -> Vec<usize> {
        (0..count).collect()
    }
}
