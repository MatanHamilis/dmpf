use std::{marker::PhantomData, u128};

use crate::{
    prg::double_prg_many,
    rb_okvs::{EncodedOkvs, EpsilonPercent, OkvsValue},
    EmptySession,
};
use rand::{CryptoRng, RngCore};

use crate::{
    prg::{double_prg, DOUBLE_PRG_CHILDREN},
    random_u126, random_u128,
    trie::BinaryTrie,
    utils::BitSlice,
    utils::BitVec,
    utils::Node,
    Dmpf, DmpfKey, DpfOutput, BITS_OF_SECURITY,
};

#[derive(Clone)]
pub enum Okvs<const W: usize, Output: OkvsValue> {
    InformationTheoretic(InformationTheoreticOkvs<Output>),
    RbOkvs(EncodedOkvs<W, Node, Output>),
}
impl<const W: usize, Output: OkvsValue> Okvs<W, Output> {
    fn decode(&self, key: &u128) -> Output {
        match self {
            Okvs::RbOkvs(v) => v.decode(&Node::from(*key)),
            Okvs::InformationTheoretic(v) => v.decode(key),
        }
    }
}
pub struct OkvsDmpfKey<const W: usize, Output: DpfOutput> {
    seed: Node,
    sign: bool,
    cws: Vec<Okvs<W, Node>>,
    last_cw: Okvs<W, Output>,
    _p: PhantomData<Output>,
}

impl<const W: usize, Output: DpfOutput> DmpfKey<Output> for OkvsDmpfKey<W, Output> {
    type Session = EmptySession;
    fn eval_with_session(&self, input: &u128, output: &mut Output, session: Self::Session) {
        let input_length = self.cws.len();
        let mut seed = self.seed;
        let mut sign = self.sign;
        let input_node = Node::from(*input);
        for i in 0..input_length {
            let mut v = input_node;
            v.mask(i);
            let mut correction_seed = OkvsDmpf::<W, Output>::correct(v, sign, &self.cws[i]);
            let (correction_sign_left, correction_sign_right) =
                correction_seed.pop_first_two_bits();
            let seeds = double_prg(&seed, &DOUBLE_PRG_CHILDREN);
            let signs = [correction_sign_left, correction_sign_right];
            let input_bit = input_node.get_bit(i);
            let input_bit_usize = input_bit as usize;
            let mut seed_prg = seeds[input_bit_usize];
            let (sign_prg, _) = seed_prg.pop_first_two_bits();
            seed = &correction_seed ^ &seed_prg;
            sign = signs[input_bit_usize] ^ sign_prg;
        }
        let mut node_output = Output::from(seed);
        node_output += OkvsDmpf::conv_correct(input_node, sign, &self.last_cw);
        // node_output.mask(self.output_len);
        *output = if self.sign {
            node_output.neg()
        } else {
            node_output
        }
    }
    fn eval_all_with_session(&self, session: Self::Session) -> Vec<Output> {
        let input_length = self.cws.len();
        let mut sign = vec![self.sign];
        let mut seed = vec![self.seed];
        for i in 0..input_length {
            let mut next_sign = Vec::with_capacity(1 << (i + 1));
            unsafe { next_sign.set_len(1 << (i + 1)) };
            let mut next_seed = Vec::with_capacity(1 << (i + 1));
            unsafe { next_seed.set_len(1 << (i + 1)) };
            double_prg_many(&seed, &DOUBLE_PRG_CHILDREN, &mut next_seed);
            let bits_left = (BITS_OF_SECURITY - i) & (BITS_OF_SECURITY - 1);
            next_seed
                .chunks_exact_mut(2)
                .zip(next_sign.chunks_exact_mut(2))
                .enumerate()
                .for_each(|(k, (seeds, signs))| {
                    // }
                    // for k in 0..1 << i {
                    let current_node = (k as u128) << bits_left;
                    let mut correction_seed =
                        OkvsDmpf::<W, Output>::correct(current_node.into(), sign[k], &self.cws[i]);
                    let (correction_sign_left, correction_sign_right) =
                        correction_seed.pop_first_two_bits();
                    let mut seed_prg_false = seeds[0];
                    let mut seed_prg_true = seeds[1];
                    let [sign_corr_false, sign_corr_true] =
                        [correction_sign_left, correction_sign_right];
                    let (sign_prg_false, _) = seed_prg_false.pop_first_two_bits();
                    let (sign_prg_true, _) = seed_prg_true.pop_first_two_bits();
                    let seed_false = &correction_seed ^ &seed_prg_false;
                    let seed_true = &correction_seed ^ &seed_prg_true;
                    let sign_false = sign_corr_false ^ sign_prg_false;
                    let sign_true = sign_corr_true ^ sign_prg_true;
                    seeds[0] = seed_false;
                    seeds[1] = seed_true;
                    signs[0] = sign_false;
                    signs[1] = sign_true;
                    // next_seed.push(seed_false);
                    // next_seed.push(seed_true);
                    // next_sign.push(sign_false);
                    // next_sign.push(sign_true);
                });
            seed = next_seed;
            sign = next_sign;
        }
        let last_cw = &self.last_cw;
        seed.into_iter()
            .zip(sign.into_iter())
            .enumerate()
            .map(|(idx, (seed, sign))| {
                let input_node = ((idx as u128) << (BITS_OF_SECURITY - input_length)).into();
                let mut node_output = Output::from(seed);
                node_output += OkvsDmpf::conv_correct(input_node, sign, last_cw);
                // node_output.mask(self.output_len);
                if self.sign {
                    node_output.neg()
                } else {
                    node_output
                }
            })
            .collect::<Vec<_>>()
            .into()
    }
    fn make_session(&self) -> Self::Session {
        EmptySession
    }
}

pub struct OkvsDmpf<const W: usize, Output: DpfOutput> {
    epsilon_percent: EpsilonPercent,
    _phantom: PhantomData<Output>,
}
impl<const W: usize, Output: DpfOutput> OkvsDmpf<W, Output> {
    pub fn new(epsilon_percent: EpsilonPercent) -> Self {
        Self {
            epsilon_percent,
            _phantom: PhantomData,
        }
    }
}

impl<const W: usize, Output: DpfOutput + OkvsValue> Dmpf<Output> for OkvsDmpf<W, Output> {
    type Key = OkvsDmpfKey<W, Output>;
    fn try_gen<R: CryptoRng + RngCore>(
        &self,
        input_length: usize,
        points: &[(u128, Output)],
        rng: &mut R,
    ) -> Option<(Self::Key, Self::Key)> {
        for i in 0..points.len() - 1 {
            assert!(points[i].0 < points[i + 1].0);
        }
        // if points.len() != self.point_count {
        //     return None;
        // }
        let t = points.len();
        // assert all inputs and all outputs are of the same length.
        let input_len = input_length;

        // We initialize the trie with the points.
        let mut trie = BinaryTrie::default();
        for point in points {
            trie.insert(&BitSlice::new(
                input_len,
                &std::slice::from_ref(&Node::from(point.0)),
            ));
        }
        let [(seed_0, sign_0), (seed_1, sign_1)] = Self::initialize(rng);
        let mut signs_0 = Vec::with_capacity(t);
        let mut seeds_0 = Vec::with_capacity(t);
        signs_0.push(sign_0);
        seeds_0.push(seed_0);
        let mut signs_1 = Vec::with_capacity(t);
        let mut seeds_1 = Vec::with_capacity(t);
        signs_1.push(sign_1);
        seeds_1.push(seed_1);
        let mut cws = Vec::with_capacity(input_len);
        for depth in 0..input_len {
            let mut next_signs_0 = Vec::with_capacity(t);
            let mut next_seeds_0 = Vec::with_capacity(t);
            let mut next_signs_1 = Vec::with_capacity(t);
            let mut next_seeds_1 = Vec::with_capacity(t);
            let mut str_bitvec = BitVec::new(depth.max(1));
            let cur_cw = Self::gen_cw(
                depth,
                &trie,
                &seeds_0,
                &signs_0,
                &seeds_1,
                &signs_1,
                self.epsilon_percent,
                rng,
            );
            let mut trie_iter = trie.iter_at_depth(depth);
            let mut idx = 0;
            loop {
                let node = match trie_iter.next() {
                    None => break,
                    Some(v) => v,
                };

                trie_iter.obtain_string(&mut str_bitvec);
                // Correcting first share
                let mut first_cw = Self::correct(str_bitvec.as_ref()[0], signs_0[idx], &cur_cw);
                let (first_sign_left, first_sign_right) = first_cw.pop_first_two_bits();
                let first_signs = [first_sign_left, first_sign_right];

                // Correcting second share
                let mut second_cw = Self::correct(str_bitvec.as_ref()[0], signs_1[idx], &cur_cw);
                let (second_sign_left, second_sign_right) = second_cw.pop_first_two_bits();
                let second_signs = [second_sign_left, second_sign_right];

                let prg_0 = double_prg(&seeds_0[idx], &DOUBLE_PRG_CHILDREN);
                let prg_1 = double_prg(&seeds_1[idx], &DOUBLE_PRG_CHILDREN);
                for son_direction in 0..=1 {
                    if node.borrow().get_son(son_direction != 0).is_some() {
                        let mut cur_0 = prg_0[son_direction];
                        let mut cur_1 = prg_1[son_direction];
                        let (cur_bit_0, _) = cur_0.pop_first_two_bits();
                        let (cur_bit_1, _) = cur_1.pop_first_two_bits();
                        next_seeds_0.push(&cur_0 ^ &first_cw);
                        next_seeds_1.push(&cur_1 ^ &second_cw);
                        next_signs_0.push(cur_bit_0 ^ first_signs[son_direction]);
                        next_signs_1.push(cur_bit_1 ^ second_signs[son_direction]);
                    } else {
                        let mut cur_0 = prg_0[son_direction];
                        let mut cur_1 = prg_1[son_direction];
                        let (cur_bit_0, _) = cur_0.pop_first_two_bits();
                        let (cur_bit_1, _) = cur_1.pop_first_two_bits();
                        assert_eq!(&cur_0 ^ &first_cw, &cur_1 ^ &second_cw);
                        assert_eq!(
                            cur_bit_0 ^ first_signs[son_direction],
                            cur_bit_1 ^ second_signs[son_direction]
                        );
                    }
                }
                idx += 1;
            }
            cws.push(cur_cw);
            (signs_0, seeds_0, signs_1, seeds_1) =
                (next_signs_0, next_seeds_0, next_signs_1, next_seeds_1);
        }
        let last_cw = Okvs::<W, Output>::RbOkvs(Self::gen_conv_cw(
            points,
            &seeds_0,
            &signs_0,
            &seeds_1,
            &signs_1,
            self.epsilon_percent,
        ));
        let first_key = OkvsDmpfKey::<W, Output> {
            seed: seed_0,
            sign: sign_0,
            cws: cws.clone(),
            last_cw: last_cw.clone(),
            _p: PhantomData,
        };
        let second_key = OkvsDmpfKey::<W, Output> {
            seed: seed_1,
            sign: sign_1,
            cws,
            last_cw,
            _p: PhantomData,
        };
        Some((first_key, second_key))
    }
}

impl<const W: usize, Output: DpfOutput> OkvsDmpf<W, Output> {
    fn initialize<R: RngCore + CryptoRng>(rng: &mut R) -> [(Node, bool); 2] {
        let output = core::array::from_fn(|i| {
            let mut word: Node = random_u128(rng).into();
            let (_, _) = word.pop_first_two_bits();
            // Notice: LAMBDA = 126.
            (word, i == 1)
        });
        output
    }
    fn correct(v: Node, sign: bool, cw: &Okvs<W, Node>) -> Node {
        if !sign {
            return Node::default();
        }
        cw.decode(&v.into()).into()
    }
    fn conv_correct(v: Node, sign: bool, cw: &Okvs<W, Output>) -> Output {
        if sign {
            cw.decode(&v.into())
        } else {
            Output::default()
        }
    }
    fn gen_conv_cw(
        kvs: &[(u128, Output)],
        seeds_0: &[Node],
        signs_0: &[bool],
        seeds_1: &[Node],
        signs_1: &[bool],
        epsilon_percent: EpsilonPercent,
    ) -> EncodedOkvs<W, Node, Output> {
        let v: Vec<_> = (0..kvs.len())
            .map(|k| {
                let out_0 = Output::from(seeds_0[k]);
                let out_1 = Output::from(seeds_1[k]);
                // let delta_g = out_0 - out_1;
                let mut cw = out_0 - out_1 - kvs[k].1;
                if signs_0[k] {
                    cw = -cw;
                }
                (Node::from(kvs[k].0), cw)
            })
            .collect();
        crate::rb_okvs::encode(&v, epsilon_percent)
    }
    fn gen_cw<R: RngCore + CryptoRng>(
        depth: usize,
        points_trie: &BinaryTrie,
        seed_0: &[Node],
        sign_0: &[bool],
        seed_1: &[Node],
        sign_1: &[bool],
        epsilon_percent: EpsilonPercent,
        rng: &mut R,
    ) -> Okvs<W, Node> {
        if depth >= BITS_OF_SECURITY {
            unimplemented!();
        }
        let t = points_trie.len();
        debug_assert_eq!(seed_0.len(), seed_1.len());
        debug_assert_eq!(seed_0.len(), sign_0.len());
        debug_assert_eq!(seed_0.len(), sign_1.len());
        let mut trie_iterator = points_trie.iter_at_depth(depth);
        let mut idx = 0;
        let mut v: Vec<(u128, u128)> = Vec::new();
        loop {
            let node = match trie_iterator.next() {
                None => break,
                Some(v) => v,
            };
            let mut str = BitVec::new(depth.max(1));
            trie_iterator.obtain_string(&mut str);
            let [mut left_0, mut right_0] = double_prg(&seed_0[idx], &DOUBLE_PRG_CHILDREN);
            let (left_sign_0, _) = left_0.pop_first_two_bits();
            let (right_sign_0, _) = right_0.pop_first_two_bits();
            let [mut left_1, mut right_1] = double_prg(&seed_1[idx], &DOUBLE_PRG_CHILDREN);
            let (left_sign_1, _) = left_1.pop_first_two_bits();
            let (right_sign_1, _) = right_1.pop_first_two_bits();
            let delta_seed_left = &left_0 ^ &left_1;
            let delta_seed_right = &right_0 ^ &right_1;
            let delta_sign_left = left_sign_0 ^ left_sign_1;
            let delta_sign_right = right_sign_0 ^ right_sign_1;
            let (left_son, right_son) = {
                let node_borrow = node.borrow();
                (node_borrow.get_son(false), node_borrow.get_son(true))
            };
            let r = if left_son.is_some() && right_son.is_some() {
                let mut r: Node = random_u126(rng).into();
                r.push_first_two_bits(!delta_sign_left, !delta_sign_right);
                r
            } else {
                debug_assert!(left_son.is_some() || right_son.is_some());
                let (mut r, left_sign, right_sign) = if left_son.is_some() {
                    (delta_seed_right, !delta_sign_left, delta_sign_right)
                } else {
                    (delta_seed_left, delta_sign_left, !delta_sign_right)
                };
                r.push_first_two_bits(left_sign, right_sign);
                r
            };
            v.push((str.as_ref()[0].into(), u128::from(r)));
            idx += 1;
        }
        let mut candidate = 0;
        let step = 1u128.overflowing_shl((BITS_OF_SECURITY - depth) as u32).0;
        let mut v_idx = 0;
        let v_orig_len = v.len();
        let new_vec_len = t.min(1 << (depth.min((usize::BITS - 1) as usize)));
        let items_to_add = new_vec_len - v.len();
        let new_v = if items_to_add == 0 {
            v
        } else {
            let mut new_v = Vec::with_capacity(new_vec_len);
            let mut items_added = 0;
            while new_v.len() < new_vec_len && items_added < items_to_add {
                if v_idx < v_orig_len {
                    if v[v_idx].0 == candidate {
                        new_v.push(v[v_idx]);
                        candidate += step;
                        v_idx += 1;
                        continue;
                    }
                }
                items_added += 1;
                let val = random_u128(rng);
                new_v.push((candidate, val));
                candidate = candidate.overflowing_add(step).0;
            }
            while new_v.len() < new_vec_len {
                new_v.push(v[v_idx]);
                v_idx += 1;
            }
            new_v
        };
        // In this case we go for information theoretic OKVS
        if points_trie.len() >= (1 << depth) {
            Okvs::InformationTheoretic(InformationTheoreticOkvs::encode(
                depth,
                new_v.into_iter().map(|v| v.1.into()).collect(),
            ))
        } else {
            let v: Vec<_> = new_v
                .into_iter()
                .map(|(a, b)| (a.into(), b.into()))
                .collect();
            Okvs::RbOkvs(crate::rb_okvs::encode::<W, _, _>(&v, epsilon_percent))
        }
    }
}

#[derive(Clone)]
pub struct InformationTheoreticOkvs<Output: OkvsValue>(Box<[Output]>, usize);
impl<Output: OkvsValue> InformationTheoreticOkvs<Output> {
    fn encode(mut input_length_in_bits: usize, values: Box<[Output]>) -> Self {
        assert_eq!(values.len() as u128, 1u128 << input_length_in_bits);
        if input_length_in_bits == 0 {
            input_length_in_bits = 128
        }
        Self(values, input_length_in_bits)
    }
    fn decode(&self, idx: &u128) -> Output {
        // Since this is information theoretic, the size can't be too big.
        // Therefore, we consider only the first u64 in i.
        self.0[(idx >> (128 - self.1)) as usize]
    }
}

#[cfg(test)]
mod test {
    use super::OkvsDmpf;
    use crate::{Dmpf, DmpfKey, Node};
    use rand::{thread_rng, RngCore};
    use std::collections::HashMap;

    #[test]
    fn test_okvs_dmpf() {
        const W: usize = 400;
        const POINTS: usize = 30;
        const INPUT_SIZE: usize = 9;
        let scheme = OkvsDmpf::<W, Node>::new(crate::rb_okvs::EpsilonPercent::Ten);
        let mut rng = thread_rng();
        let mut input_map = HashMap::with_capacity(POINTS);
        while input_map.len() < POINTS {
            let i = (rng.next_u64() % (1 << INPUT_SIZE)) as usize;
            let encoded_i = encode_input(i, INPUT_SIZE);
            if input_map.contains_key(&encoded_i) {
                continue;
            }
            let random_v = Node::random(&mut rng);
            input_map.insert(encoded_i, random_v);
        }
        let mut inputs: Vec<_> = input_map.iter().map(|(&a, &b)| (a, b)).collect();
        inputs.sort();
        let (key_1, key_2) = scheme.try_gen(INPUT_SIZE, &inputs[..], &mut rng).unwrap();
        let eval_all_1 = key_1.eval_all();
        let eval_all_2 = key_2.eval_all();
        let eval_all_sum: Vec<_> = eval_all_1
            .iter()
            .zip(eval_all_2.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        for i in 0..(1 << INPUT_SIZE) {
            let mut output_1 = Node::default();
            let mut output_2 = Node::default();
            let encoded_i = encode_input(i, INPUT_SIZE);
            key_1.eval(&encoded_i, &mut output_1);
            key_2.eval(&encoded_i, &mut output_2);
            assert_eq!(output_1, eval_all_1[i]);
            assert_eq!(output_2, eval_all_2[i]);
            let output = output_1 + output_2;
            if input_map.contains_key(&encoded_i) {
                assert_eq!(output, input_map[&encoded_i]);
            } else {
                assert_eq!(output, Node::default());
            };
        }
    }
    fn encode_input(i: usize, input_len: usize) -> u128 {
        (i as u128) << (128 - input_len)
    }
}
