use std::collections::{HashMap, HashSet};

use aes_prng::AesRng;
use rand::{CryptoRng, RngCore, SeedableRng};
use rb_okvs::OkvsValueU128Array;

use crate::{
    utils::BitSlice, utils::BitSliceMut, utils::BitVec, utils::Node, Dmpf, DmpfKey, DpfKey,
};

pub struct BatchCodeDmpf<const WIDTH: usize> {
    hash_functions_count: usize,
    expansion_overhead_in_percent: usize,
}

impl<const WIDTH: usize> BatchCodeDmpf<WIDTH> {
    pub fn new(hash_functions_count: usize, expansion_overhead_in_percent: usize) -> Self {
        Self {
            hash_functions_count,
            expansion_overhead_in_percent,
        }
    }
}

pub struct BatchCodeDmpfKey<const WIDTH: usize> {
    input_domain_log_size: usize,
    dpf_input_length: usize,
    buckets: Vec<DpfKey<WIDTH>>,
    hash_functions: HashFunctionIndex,
}

impl<const WIDTH: usize> Dmpf for BatchCodeDmpf<WIDTH> {
    type Key = BatchCodeDmpfKey<WIDTH>;
    fn try_gen<R: rand::prelude::CryptoRng + rand::prelude::RngCore>(
        &self,
        input_length: usize,
        inputs: &[(
            <Self::Key as DmpfKey>::InputContainer,
            <Self::Key as DmpfKey>::OutputContainer,
        )],
        mut rng: &mut R,
    ) -> Option<(Self::Key, Self::Key)> {
        let buckets = (inputs.len() * (100 + self.expansion_overhead_in_percent)) / 100;
        let bucket_size = ((1 << input_length) * self.hash_functions_count).div_ceil(buckets);
        let dpf_input_length = usize::ilog2(2 * bucket_size - 1) as usize;
        // assert_eq!(1 << bucket_log_size, bucket_size);
        let index_to_value_map: HashMap<_, _> = inputs
            .iter()
            .map(|v| {
                (
                    (v.0[0] >> ((u128::BITS as usize) - input_length)) as usize,
                    v.1,
                )
            })
            .collect();
        let indices: Vec<_> = index_to_value_map.keys().copied().collect();
        let mut hash_functions_seed = [0u8; aes_prng::SEED_SIZE];
        rng.fill_bytes(&mut hash_functions_seed);
        let encoding = batch_encode::<AesRng>(
            input_length,
            &indices[..],
            buckets,
            self.hash_functions_count,
            hash_functions_seed,
            &mut rng,
        );
        let mut dpfs: Vec<_> = (0..buckets)
            .map(|_| {
                let roots = (Node::random(&mut rng), Node::random(&mut rng));
                let alpha = OkvsValueU128Array::default();
                let beta = OkvsValueU128Array::default();
                DpfKey::gen(&roots, &alpha, dpf_input_length, &beta)
            })
            .collect();
        for (index, (bucket, index_in_bucket)) in encoding {
            let value = index_to_value_map[&index];
            let mut alpha = OkvsValueU128Array::default();
            alpha[0] = ((index_in_bucket as u128) << (128 - dpf_input_length)).into();
            let beta = value;
            let roots = (Node::random(&mut rng), Node::random(&mut rng));
            dpfs[bucket] = DpfKey::gen(&roots, &alpha, dpf_input_length, &beta);
        }
        let (dpfs_0, dpfs_1): (Vec<_>, Vec<_>) = dpfs.into_iter().unzip();
        let (_, hash_functions) = gen_hash_functions::<AesRng>(
            input_length,
            bucket_size,
            self.hash_functions_count,
            hash_functions_seed,
        );
        Some((
            BatchCodeDmpfKey {
                input_domain_log_size: input_length,
                dpf_input_length,
                buckets: dpfs_0,
                hash_functions: hash_functions.clone(),
            },
            BatchCodeDmpfKey {
                input_domain_log_size: input_length,
                dpf_input_length,
                buckets: dpfs_1,
                hash_functions,
            },
        ))
    }
}

impl<const WIDTH: usize> DmpfKey for BatchCodeDmpfKey<WIDTH> {
    type InputContainer = OkvsValueU128Array<1>;
    type OutputContainer = OkvsValueU128Array<WIDTH>;
    type Session = ();
    fn eval(&self, input: &Self::InputContainer, output: &mut Self::OutputContainer) {
        // Zero output buffer.
        *output ^= *output;
        let input_usize = (input[0] >> (128 - self.input_domain_log_size)) as usize;
        // let mut output_node = [Node::default(); WIDTH];
        // let mut output_slice = BitSliceMut::new(128, &mut output_node);

        for f in self.hash_functions.iter() {
            let (bucket, index) = self.hash_functions.eval(input_usize, f);
            let dpf_input =
                OkvsValueU128Array::from([((index as u128) << (128 - self.dpf_input_length))]);
            // let input_bitvec = BitSlice::new(self.dpf_input_length, &node[..]);
            let mut cur_output = [Node::default(); WIDTH];
            self.buckets[bucket].eval(&dpf_input, &mut cur_output);
            *output = core::array::from_fn(|i| output[i] ^ u128::from(cur_output[i])).into();
        }
    }
    fn eval_all(&self) -> Box<[Self::OutputContainer]> {
        let dpfs_eval_all: Vec<_> = self.buckets.iter().map(|dpf| dpf.eval_all()).collect();
        let input_domain_size = 1 << self.input_domain_log_size;
        let mut output = Vec::with_capacity(input_domain_size);
        for i in 0..input_domain_size {
            let mut output_cur = OkvsValueU128Array::<WIDTH>::default();
            for (bucket, bucket_idx) in self.hash_functions.eval_all(i) {
                let c = dpfs_eval_all[bucket][bucket_idx];
                for j in 0..WIDTH {
                    output_cur[j] ^= u128::from(c[j]);
                }
            }
            output.push(output_cur);
        }
        output.into()
    }
    fn make_session(&self) -> Self {
        unimplemented!()
    }
}

#[derive(Clone)]
struct HashFunctionIndex {
    contents: Vec<usize>,
    hash_functions_count: usize,
    bucket_size: usize,
}
impl HashFunctionIndex {
    fn new(hash_functions_count: usize, input_domain_log_size: usize, bucket_size: usize) -> Self {
        Self {
            contents: vec![usize::MAX; hash_functions_count << input_domain_log_size],
            hash_functions_count,
            bucket_size,
        }
    }
    fn map(&mut self, from: usize, to: usize) {
        let base_idx = from * self.hash_functions_count;
        for i in 0..self.hash_functions_count {
            if self.contents[base_idx + i] == usize::MAX {
                self.contents[base_idx + i] = to;
                return;
            }
        }
        panic!("Too many mappings");
    }
    fn eval(&self, from: usize, function_idx: usize) -> (usize, usize) {
        debug_assert!(function_idx < self.hash_functions_count);
        let v = self.contents[self.hash_functions_count * from + function_idx];
        (v / self.bucket_size, v % self.bucket_size)
    }
    fn eval_all(&self, from: usize) -> impl '_ + Iterator<Item = (usize, usize)> {
        let v = &self.contents
            [self.hash_functions_count * from..self.hash_functions_count * (from + 1)];
        v.iter()
            .map(|v| (v / self.bucket_size, v % self.bucket_size))
    }
    fn iter(&self) -> impl Iterator<Item = usize> {
        0..self.hash_functions_count
    }
}

fn gen_hash_functions<R: RngCore + CryptoRng + SeedableRng>(
    input_domain_log_size: usize,
    bucket_size: usize,
    hash_functions_count: usize,
    seed: R::Seed,
) -> (Vec<usize>, HashFunctionIndex) {
    let mut rng = R::from_seed(seed);
    let total_domain_size = hash_functions_count << input_domain_log_size;
    let mut permutation: Vec<_> = (0..(1 << input_domain_log_size))
        .cycle()
        .take(total_domain_size)
        .collect();
    let mut mapping =
        HashFunctionIndex::new(hash_functions_count, input_domain_log_size, bucket_size);
    for i in 0..permutation.len() {
        let swap_id = i + (rng.next_u64() as usize) % (permutation.len() - i);
        permutation.swap(i, swap_id);
        let from_mapping = permutation[i];
        mapping.map(from_mapping, i);
    }
    (permutation, mapping)
}

fn batch_encode<R: CryptoRng + RngCore + SeedableRng>(
    input_domain_log_size: usize,
    indices: &[usize],
    buckets: usize,
    hash_functions_count: usize,
    seed: R::Seed,
    mut rng: impl CryptoRng + RngCore,
) -> Vec<(usize, (usize, usize))>
where
    R::Seed: Clone + Copy,
{
    let input_domain_size = 1 << input_domain_log_size;
    let bucket_size = (input_domain_size * hash_functions_count).div_ceil(buckets);
    let (_, hash_functions) = gen_hash_functions::<R>(
        input_domain_log_size,
        bucket_size,
        hash_functions_count,
        seed,
    );
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
        let function_id = bucket_randomness % hash_functions_count;
        bucket_randomness /= hash_functions_count;
        if bucket_randomness < hash_functions_count {
            bucket_randomness = rng.next_u64() as usize;
        }

        let (bucket, index_in_bucket) = hash_functions.eval(item_to_map, function_id);

        index_to_map = if let Some(v) = bucket_to_item[bucket].replace(index_to_map) {
            item_to_bucket[index_to_map] = Some((bucket, index_in_bucket));
            item_to_bucket[v] = None;
            iterations_without_advance += 1;
            if iterations_without_advance <= indices.len() * hash_functions_count {
                v
            } else {
                panic!("Can't find matching, buckets count: {}", buckets);
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
    use std::collections::{HashMap, HashSet};

    use aes_prng::AesRng;
    use rand::{thread_rng, RngCore};
    use rb_okvs::OkvsValueU128Array;

    use crate::{
        batch_code::{gen_hash_functions, BatchCodeDmpf},
        random_u128, Dmpf, DmpfKey,
    };

    use super::batch_encode;

    #[test]
    fn test_batch_code() {
        let input_domain_log_size: usize = 15;
        let input_domain_size = 1 << input_domain_log_size;
        let indices_count: usize = 1 << 10;
        let hash_functions_count = 3;
        let indices = make_indices(indices_count);
        let mut seed = [0u8; aes_prng::SEED_SIZE];
        thread_rng().fill_bytes(&mut seed);
        let buckets = indices_count * 3 / 2;
        let bucket_size = ((input_domain_size * hash_functions_count) as usize).div_ceil(buckets);
        let encoding = batch_encode::<AesRng>(
            input_domain_log_size,
            &indices[..],
            buckets,
            hash_functions_count,
            seed,
            thread_rng(),
        );

        // Ensure we receive a mapping for each input.
        assert_eq!(encoding.len(), indices.len());

        let (_, hash_functions) = gen_hash_functions::<AesRng>(
            input_domain_log_size,
            bucket_size,
            hash_functions_count,
            seed,
        );

        // Assert all buckets a output of some hash function for the given index.
        for i in indices.iter().copied() {
            let (bucket, index_in_bucket) = encoding
                .iter()
                .copied()
                .find_map(|v| if v.0 == i { Some(v.1) } else { None })
                .unwrap();
            assert!((0..hash_functions_count).any(|f| hash_functions.eval(i, f).0 == bucket));
            assert!(index_in_bucket < bucket_size);
        }

        // Asserts all buckets are distinct.
        let set: HashSet<_> = encoding.iter().map(|(_, b)| *b).collect();
        assert_eq!(set.len(), indices.len());
    }
    fn make_indices(count: usize) -> Vec<usize> {
        (0..count).collect()
    }

    #[test]
    fn test_batch_code_dmpf() {
        const W: usize = 4;
        const POINTS: usize = 10;
        const INPUT_SIZE: usize = 4;
        const OUTPUT_SIZE: usize = 127;
        const EXPANSION_OVERHEAD_IN_PERCENT: usize = 50;
        let scheme = BatchCodeDmpf::new(W, EXPANSION_OVERHEAD_IN_PERCENT);
        let mut rng = thread_rng();
        let output_mask: u128 = ((1u128 << OUTPUT_SIZE) - 1) << (128 - OUTPUT_SIZE);
        let inputs: [_; POINTS] = core::array::from_fn(|i| {
            (
                encode_input(i, INPUT_SIZE),
                OkvsValueU128Array::from([random_u128(&mut rng) & output_mask]),
            )
        });
        let input_map: HashMap<_, _> = inputs.iter().copied().collect();
        let (key_1, key_2) = scheme.try_gen(INPUT_SIZE, &inputs[..], &mut rng).unwrap();
        let eval_all_1 = key_1.eval_all();
        let eval_all_2 = key_2.eval_all();
        for i in 0..(1 << INPUT_SIZE) {
            let mut output_1 = OkvsValueU128Array::default();
            let mut output_2 = OkvsValueU128Array::default();
            let encoded_i = encode_input(i, INPUT_SIZE);
            key_1.eval(&encoded_i, &mut output_1);
            key_2.eval(&encoded_i, &mut output_2);
            assert_eq!(output_1, eval_all_1[i]);
            assert_eq!(output_2, eval_all_2[i]);
            let output = output_1 ^ output_2;
            if input_map.contains_key(&encoded_i) {
                assert_eq!(output, input_map[&encoded_i]);
            } else {
                assert_eq!(output, OkvsValueU128Array::default());
            };
        }
    }
    fn encode_input(i: usize, input_len: usize) -> OkvsValueU128Array<1> {
        OkvsValueU128Array::from([(i as u128) << (128 - input_len)])
    }
}
