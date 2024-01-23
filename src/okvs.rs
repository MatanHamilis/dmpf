use std::{ops::DerefMut, u128};

use rand::{CryptoRng, RngCore};
use rb_okvs::{EncodedOkvs, EpsilonPercent, OkvsValueU128Array};

use crate::{
    prg::{double_prg, many_prg, DOUBLE_PRG_CHILDREN},
    random_u126, random_u128,
    trie::BinaryTrie,
    utils::BitSlice,
    utils::BitVec,
    utils::Node,
    Dmpf, DmpfKey, BITS_OF_SECURITY,
};

#[derive(Clone)]
pub enum Okvs<const W: usize, const OKVS_WIDTH: usize> {
    InformationTheoretic(InformationTheoreticOkvs<OKVS_WIDTH>),
    RbOkvs(EncodedOkvs<W, OkvsValueU128Array<1>, OkvsValueU128Array<OKVS_WIDTH>>),
}
impl<const W: usize, const OKVS_WIDTH: usize> Okvs<W, OKVS_WIDTH> {
    fn decode(&self, key: &OkvsValueU128Array<1>) -> OkvsValueU128Array<OKVS_WIDTH> {
        match self {
            Okvs::RbOkvs(v) => v.decode(key),
            Okvs::InformationTheoretic(v) => v.decode(key),
        }
    }
}
pub struct OkvsDmpfKey<const W: usize, const OKVS_WIDTH: usize> {
    seed: Node,
    sign: bool,
    output_len: usize,
    cws: Vec<Okvs<W, 1>>,
    last_cw: Okvs<W, OKVS_WIDTH>,
}

impl From<OkvsValueU128Array<1>> for Node {
    fn from(value: OkvsValueU128Array<1>) -> Self {
        Node::from(value[0])
    }
}
impl From<Node> for OkvsValueU128Array<1> {
    fn from(value: Node) -> Self {
        let o: u128 = value.into();
        o.into()
    }
}

impl<const W: usize, const OKVS_WIDTH: usize> DmpfKey for OkvsDmpfKey<W, OKVS_WIDTH> {
    type Session = ();
    type InputContainer = OkvsValueU128Array<1>;
    type OutputContainer = OkvsValueU128Array<OKVS_WIDTH>;
    fn eval(&self, input: &Self::InputContainer, output: &mut Self::OutputContainer) {
        let input_length = self.cws.len();
        let mut seed = self.seed;
        let mut sign = self.sign;
        let input_node = Node::from(*input);
        for i in 0..input_length {
            let mut v = input_node;
            v.mask(i);
            let mut correction_seed = OkvsDmpf::<W, OKVS_WIDTH>::correct(v, sign, &self.cws[i]);
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
        let mut node_output = expand_to_okvs_value::<OKVS_WIDTH>(&seed);
        node_output ^= OkvsDmpf::conv_correct(input_node, sign, &self.last_cw);
        // node_output.mask(self.output_len);
        *output = node_output;
    }
    fn eval_all(&self) -> Box<[Self::OutputContainer]> {
        let input_length = self.cws.len();
        let mut sign = vec![self.sign];
        let mut seed = vec![self.seed];
        for i in 0..input_length {
            let mut next_sign = Vec::with_capacity(1 << (i + 1));
            let mut next_seed = Vec::with_capacity(1 << (i + 1));
            let bits_left = (BITS_OF_SECURITY - i) & (BITS_OF_SECURITY - 1);
            for k in 0..1 << i {
                let current_node = (k as u128) << bits_left;
                let mut correction_seed =
                    OkvsDmpf::<W, OKVS_WIDTH>::correct(current_node.into(), sign[k], &self.cws[i]);
                let (correction_sign_left, correction_sign_right) =
                    correction_seed.pop_first_two_bits();
                let [mut seed_prg_false, mut seed_prg_true] =
                    double_prg(&seed[k], &DOUBLE_PRG_CHILDREN);
                let [sign_corr_false, sign_corr_true] =
                    [correction_sign_left, correction_sign_right];
                let (sign_prg_false, _) = seed_prg_false.pop_first_two_bits();
                let (sign_prg_true, _) = seed_prg_true.pop_first_two_bits();
                let seed_false = &correction_seed ^ &seed_prg_false;
                let seed_true = &correction_seed ^ &seed_prg_true;
                let sign_false = sign_corr_false ^ sign_prg_false;
                let sign_true = sign_corr_true ^ sign_prg_true;
                next_seed.push(seed_false);
                next_seed.push(seed_true);
                next_sign.push(sign_false);
                next_sign.push(sign_true);
            }
            seed = next_seed;
            sign = next_sign;
        }
        let last_cw = &self.last_cw;
        seed.into_iter()
            .zip(sign.into_iter())
            .enumerate()
            .map(|(idx, (seed, sign))| {
                let input_node = ((idx as u128) << (BITS_OF_SECURITY - input_length)).into();
                let mut node_output = expand_to_okvs_value(&seed);
                node_output ^= OkvsDmpf::conv_correct(input_node, sign, last_cw);
                // node_output.mask(self.output_len);
                node_output
            })
            .collect::<Vec<_>>()
            .into()
    }
    fn make_session(&self) -> Self {
        unimplemented!()
    }
}

pub fn expand_to_okvs_value<const OKVS_WIDTH: usize>(
    node: &Node,
) -> OkvsValueU128Array<OKVS_WIDTH> {
    let mut output = OkvsValueU128Array::default();
    if OKVS_WIDTH == 1 {
        output[0] = u128::from(*node);
    } else {
        let out_arr: &mut [u128; OKVS_WIDTH] = output.deref_mut();
        let out_node = unsafe {
            std::slice::from_raw_parts_mut(out_arr.as_mut_ptr() as *mut Node, OKVS_WIDTH / 2)
        };
        many_prg(node, 0..(out_node.len() as u16), out_node);
    }
    output
}
pub struct OkvsDmpf<const W: usize, const OKVS_WIDTH: usize> {
    input_len: usize,
    output_len: usize,
    point_count: usize,
    epsilon_percent: EpsilonPercent,
}
impl<const W: usize, const OKVS_WIDTH: usize> OkvsDmpf<W, OKVS_WIDTH> {
    pub fn new(
        input_len: usize,
        output_len: usize,
        point_count: usize,
        epsilon_percent: EpsilonPercent,
    ) -> Self {
        Self {
            input_len,
            output_len,
            point_count,
            epsilon_percent,
        }
    }
}

impl<const W: usize, const OKVS_WIDTH: usize> Dmpf for OkvsDmpf<W, OKVS_WIDTH> {
    type Key = OkvsDmpfKey<W, OKVS_WIDTH>;
    fn try_gen<R: CryptoRng + RngCore>(
        &self,
        points: &[(
            <Self::Key as DmpfKey>::InputContainer,
            <Self::Key as DmpfKey>::OutputContainer,
        )],
        rng: &mut R,
    ) -> Option<(Self::Key, Self::Key)> {
        if points.len() != self.point_count {
            return None;
        }
        let t = points.len();
        // assert all inputs and all outputs are of the same length.
        let input_len = self.input_len;
        let output_len = self.output_len;

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
        let last_cw = Okvs::<W, OKVS_WIDTH>::RbOkvs(Self::gen_conv_cw(
            points,
            &seeds_0,
            &signs_0,
            &seeds_1,
            &signs_1,
            self.epsilon_percent,
        ));
        let first_key = OkvsDmpfKey::<W, OKVS_WIDTH> {
            seed: seed_0,
            sign: sign_0,
            output_len,
            cws: cws.clone(),
            last_cw: last_cw.clone(),
        };
        let second_key = OkvsDmpfKey::<W, OKVS_WIDTH> {
            seed: seed_1,
            sign: sign_1,
            output_len,
            cws,
            last_cw,
        };
        Some((first_key, second_key))
    }
}

impl<const W: usize, const OKVS_WIDTH: usize> OkvsDmpf<W, OKVS_WIDTH> {
    fn initialize<R: RngCore + CryptoRng>(rng: &mut R) -> [(Node, bool); 2] {
        let output = core::array::from_fn(|i| {
            let mut word: Node = random_u128(rng).into();
            let (_, _) = word.pop_first_two_bits();
            // Notice: LAMBDA = 126.
            (word, i == 1)
        });
        output
    }
    fn correct(v: Node, sign: bool, cw: &Okvs<W, 1>) -> Node {
        if !sign {
            return Node::default();
        }
        cw.decode(&v.into()).into()
    }
    fn conv_correct(
        v: Node,
        sign: bool,
        cw: &Okvs<W, OKVS_WIDTH>,
    ) -> OkvsValueU128Array<OKVS_WIDTH> {
        if sign {
            OkvsValueU128Array::<OKVS_WIDTH>::default()
        } else {
            cw.decode(&v.into())
        }
    }
    fn gen_conv_cw(
        kvs: &[(OkvsValueU128Array<1>, OkvsValueU128Array<OKVS_WIDTH>)],
        seeds_0: &[Node],
        signs_0: &[bool],
        seeds_1: &[Node],
        signs_1: &[bool],
        epsilon_percent: EpsilonPercent,
    ) -> EncodedOkvs<W, OkvsValueU128Array<1>, OkvsValueU128Array<OKVS_WIDTH>> {
        let v: Vec<_> = (0..kvs.len())
            .map(|k| {
                let out_0 = expand_to_okvs_value(&seeds_0[k]);
                let out_1 = expand_to_okvs_value(&seeds_1[k]);
                let delta_g = out_0 ^ out_1;
                (kvs[k].0, kvs[k].1 ^ delta_g)
            })
            .collect();
        rb_okvs::encode(&v, epsilon_percent)
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
    ) -> Okvs<W, 1> {
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
            Okvs::RbOkvs(rb_okvs::encode::<W, _, _>(&v, epsilon_percent))
        }
    }
}

struct Cw {
    items: Vec<u128>,
    t: usize,
}
impl Cw {
    fn random<R: CryptoRng + RngCore>(t: usize, rng: &mut R) -> Self {
        let items_per_single_cw = 1;
        let total_items = t * items_per_single_cw;
        let mut items = (0..total_items).map(|_| random_u128(rng)).collect();
        Self { items, t }
    }
}
#[derive(Clone)]
pub struct InformationTheoreticOkvs<const OKVS_WIDTH: usize>(
    Box<[OkvsValueU128Array<OKVS_WIDTH>]>,
    usize,
);
impl<const OKVS_WIDTH: usize> InformationTheoreticOkvs<OKVS_WIDTH> {
    fn encode(
        mut input_length_in_bits: usize,
        values: Box<[OkvsValueU128Array<OKVS_WIDTH>]>,
    ) -> Self {
        assert_eq!(values.len(), 1 << input_length_in_bits);
        if input_length_in_bits == 0 {
            input_length_in_bits = 64;
        }
        Self(values, input_length_in_bits)
    }
    fn decode(&self, i: &OkvsValueU128Array<1>) -> OkvsValueU128Array<OKVS_WIDTH> {
        // Since this is information theoretic, the size can't be too big.
        // Therefore, we consider only the first u64 in i.
        let idx = i[0];
        self.0[(idx >> (64 - self.1)) as usize]
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use rand::thread_rng;
    use rb_okvs::{OkvsValue, OkvsValueU128Array};

    use crate::{Dmpf, DmpfKey};

    use super::OkvsDmpf;

    #[test]
    fn test_okvs_dmpf() {
        const W: usize = 5;
        const POINTS: usize = 1;
        const INPUT_SIZE: usize = 8;
        const OKVS_WIDTH: usize = 2;
        // const OUTPUT_SIZE: usize = 10;
        let scheme = OkvsDmpf::<W, OKVS_WIDTH>::new(
            INPUT_SIZE,
            OKVS_WIDTH * 64,
            POINTS,
            rb_okvs::EpsilonPercent::Ten,
        );
        let mut rng = thread_rng();
        // let output_mask: u128 = ((1u128 << OUTPUT_SIZE) - 1) << (128 - OUTPUT_SIZE);
        let inputs: [_; POINTS] = core::array::from_fn(|i| {
            (
                encode_input::<1>(i, INPUT_SIZE),
                OkvsValueU128Array::<OKVS_WIDTH>::random(&mut rng),
            )
        });
        let input_map: HashMap<_, _> = inputs.iter().copied().collect();
        let (key_1, key_2) = scheme.try_gen(&inputs[..], &mut rng).unwrap();
        let eval_all_1 = key_1.eval_all();
        let eval_all_2 = key_2.eval_all();
        for i in 0..(1 << INPUT_SIZE) {
            let mut output_1 = OkvsValueU128Array::<OKVS_WIDTH>::default();
            let mut output_2 = OkvsValueU128Array::<OKVS_WIDTH>::default();
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
    fn encode_input<const OKVS_WIDTH: usize>(
        i: usize,
        input_len: usize,
    ) -> OkvsValueU128Array<OKVS_WIDTH> {
        let mut output = OkvsValueU128Array::<OKVS_WIDTH>::default();
        output[0] = (i as u128) << (128 - input_len);
        output
        // OkvsValueU64Array::from(i)(i as u128) << (128 - input_len)
    }
}
