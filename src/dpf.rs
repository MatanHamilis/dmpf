use std::time::Duration;
use std::time::Instant;

use super::BITS_OF_SECURITY;
use crate::prg::double_prg;
use crate::prg::double_prg_many;
use crate::prg::many_prg;
use crate::prg::DOUBLE_PRG_CHILDREN;
use crate::utils::BitVec;
use crate::utils::Node;
use crate::Dmpf;
use crate::DmpfKey;
use crate::DpfOutput;
use crate::EmptySession;

#[derive(Clone, Copy)]
pub struct CorrectionWord {
    node: Node,
}
pub struct DpfDmpf;
impl DpfDmpf {
    pub fn new() -> Self {
        Self
    }
}
impl<Output: DpfOutput> Dmpf<Output> for DpfDmpf {
    type Key = DpfDmpfKey<Output>;
    fn try_gen<R: rand::prelude::CryptoRng + rand::prelude::RngCore>(
        &self,
        input_length: usize,
        inputs: &[(u128, Output)],
        mut rng: &mut R,
    ) -> Option<(Self::Key, Self::Key)> {
        let mut first_keys = Vec::with_capacity(inputs.len());
        let mut second_keys = Vec::with_capacity(inputs.len());
        inputs.iter().for_each(|(k, v)| {
            let roots = (Node::random(&mut rng), Node::random(&mut rng));
            let (f, s) = DpfKey::gen(&roots, k, input_length, v);
            first_keys.push(f);
            second_keys.push(s);
        });
        Some((
            DpfDmpfKey {
                dpf_keys: first_keys,
            },
            DpfDmpfKey {
                dpf_keys: second_keys,
            },
        ))
    }
}
pub struct DpfDmpfKey<Output> {
    dpf_keys: Vec<DpfKey<Output>>,
}
impl<Output: DpfOutput> DmpfKey<Output> for DpfDmpfKey<Output> {
    type Session = EmptySession;
    fn eval_with_session(&self, input: &u128, output: &mut Output, session: &mut Self::Session) {
        *output = self
            .dpf_keys
            .iter()
            .map(|k| {
                let mut cur_out = Output::default();
                k.eval(input, &mut cur_out);
                cur_out
            })
            .sum();
    }
    fn point_count(&self) -> usize {
        self.dpf_keys.len()
    }
    fn eval_all_with_session(&self, session: &mut Self::Session) -> Vec<Output> {
        let mut f: Vec<Output> = self.dpf_keys[0].eval_all();
        self.dpf_keys[1..].iter().for_each(|k| {
            f.iter_mut().zip(k.eval_all_into()).for_each(|(o, i)| {
                *o += i;
            })
        });
        f
    }
}
pub struct DpfKey<Output> {
    root: Node,
    root_bit: bool,
    cws: Vec<CorrectionWord>,
    last_cw: Output,
    input_bits: usize,
}
impl CorrectionWord {
    fn new(mut node: Node, left_bit: bool, right_bit: bool) -> Self {
        node.push_first_two_bits(left_bit, right_bit);
        Self { node }
    }
}
pub(crate) fn tree_and_leaf_depth(alpha_len: usize, beta_len: usize) -> (usize, usize) {
    let max_betas_in_node = BITS_OF_SECURITY / beta_len;
    let max_leaf_depth = if max_betas_in_node > 0 {
        usize::ilog2(max_betas_in_node) as usize
    } else {
        0
    };
    let tree_depth = if max_leaf_depth > alpha_len {
        0
    } else {
        alpha_len - max_leaf_depth
    };
    let leaf_depth = alpha_len - tree_depth;
    (tree_depth, leaf_depth)
}

pub(crate) fn convert(node: &Node, bits: usize) -> BitVec {
    let mut output = BitVec::new(bits);
    convert_into(node, &mut output.as_mut());
    output
}
pub(crate) fn convert_into(node: &Node, output: &mut [Node]) {
    let len = output.len();
    if len > 1 {
        many_prg(node, 0..len as u16, output);
    } else {
        // We don't have to expand node
        output[0] = *node;
    }
}
fn get_bit(v: u128, bit_idx: usize) -> bool {
    (v >> (127 - bit_idx)) & 1 == 1
}
impl<Output: DpfOutput> DpfKey<Output> {
    pub fn gen(
        roots: &(Node, Node),
        alpha: &u128,
        input_len: usize,
        beta: &Output,
    ) -> (DpfKey<Output>, DpfKey<Output>) {
        let mut t_0 = false;
        let mut t_1 = true;
        let mut seed_0 = roots.0;
        let mut seed_1 = roots.1;
        let mut cws = Vec::with_capacity(input_len);
        for i in 0..input_len {
            let [mut seeds_l_0, mut seeds_r_0] = double_prg(&seed_0, &DOUBLE_PRG_CHILDREN);
            let [mut seeds_l_1, mut seeds_r_1] = double_prg(&seed_1, &DOUBLE_PRG_CHILDREN);
            let path_bit = get_bit(*alpha, i);
            let (t_l_0, _) = seeds_l_0.pop_first_two_bits();
            let (t_l_1, _) = seeds_l_1.pop_first_two_bits();
            let (t_r_0, _) = seeds_r_0.pop_first_two_bits();
            let (t_r_1, _) = seeds_r_1.pop_first_two_bits();
            let diff_bit_left = !(t_l_0 ^ t_l_1 ^ path_bit);
            let diff_bit_right = t_r_0 ^ t_r_1 ^ path_bit;
            let cw_node = [seeds_l_0 ^ seeds_l_1, seeds_r_0 ^ seeds_r_1][!path_bit as usize];
            seed_0 = [seeds_l_0, seeds_r_0][path_bit as usize];
            seed_1 = [seeds_l_1, seeds_r_1][path_bit as usize];
            if t_0 {
                seed_0 ^= &cw_node;
            }
            if t_1 {
                seed_1 ^= &cw_node;
            }
            (t_0, t_1) = if path_bit {
                (
                    t_r_0 ^ (t_0 & diff_bit_right),
                    t_r_1 ^ (t_1 & diff_bit_right),
                )
            } else {
                (t_l_0 ^ (t_0 & diff_bit_left), t_l_1 ^ (t_1 & diff_bit_left))
            };
            cws.push(CorrectionWord::new(cw_node, diff_bit_left, diff_bit_right));
        }
        let conv_0 = Output::from(seed_0);
        let conv_1 = Output::from(seed_1);
        let mut last_cw: Output = conv_0 - conv_1 - *beta;
        if t_0 {
            last_cw = last_cw.neg();
        }
        let first_key = DpfKey {
            root: roots.0,
            root_bit: false,
            cws: cws.clone(),
            last_cw: last_cw.into(),
            input_bits: input_len,
        };
        let second_key = DpfKey {
            root: roots.1,
            root_bit: true,
            cws,
            last_cw: last_cw.into(),
            input_bits: input_len,
        };
        (first_key, second_key)
    }
    pub fn eval(&self, x: &u128, output: &mut Output) {
        let mut t = self.root_bit;
        let mut s = self.root;
        for (idx, cw) in self.cws.iter().enumerate() {
            let path_bit = get_bit(*x, idx);
            let seeds = double_prg(&s, &DOUBLE_PRG_CHILDREN);
            s = seeds[path_bit as usize];
            let (mut new_t, _) = s.pop_first_two_bits();
            if t {
                let mut cw = cw.node;
                let (left_bit, right_bit) = cw.pop_first_two_bits();
                s ^= &cw;
                new_t ^= (left_bit & !path_bit) ^ (right_bit & path_bit);
            }
            t = new_t;
        }
        *output = Output::from(s);
        if t {
            *output += self.last_cw;
        }
        if self.root_bit {
            *output = output.neg()
        }
    }
    pub fn eval_all_into(&self) -> impl Iterator<Item = Output> {
        let mut cur_seeds = Vec::with_capacity(1 << self.input_bits);
        let mut next_seeds = Vec::with_capacity(1 << self.input_bits);
        unsafe { cur_seeds.set_len(1 << self.input_bits) };
        unsafe { next_seeds.set_len(1 << self.input_bits) };
        cur_seeds[0] = self.root;
        let mut cur_signs = Vec::with_capacity(1 << self.input_bits);
        let mut next_signs = Vec::with_capacity(1 << self.input_bits);
        unsafe { cur_signs.set_len(1 << self.input_bits) };
        unsafe { next_signs.set_len(1 << self.input_bits) };
        cur_signs[0] = self.root_bit;
        for depth in 0..self.input_bits {
            double_prg_many(
                &cur_seeds[..1 << depth],
                &DOUBLE_PRG_CHILDREN,
                &mut next_seeds[..2 << depth],
            );
            let mut cur_cw = self.cws[depth].node;
            let (cw_t_l, cw_t_r) = cur_cw.pop_first_two_bits();
            next_seeds[..2 << depth]
                .chunks_exact_mut(2)
                .zip(next_signs[..2 << depth].chunks_exact_mut(2))
                .zip(cur_signs[..1 << depth].iter().copied())
                .for_each(|((seeds, signs), cur_sign)| {
                    let mut seed_l = seeds[0];
                    let mut seed_r = seeds[1];
                    let (mut t_l, _) = seed_l.pop_first_two_bits();
                    let (mut t_r, _) = seed_r.pop_first_two_bits();
                    if cur_sign {
                        seed_l ^= &cur_cw;
                        seed_r ^= &cur_cw;
                        t_l ^= cw_t_l;
                        t_r ^= cw_t_r;
                    }
                    seeds[0] = seed_l;
                    seeds[1] = seed_r;
                    signs[0] = t_l;
                    signs[1] = t_r;
                });
            (cur_seeds, next_seeds) = (next_seeds, cur_seeds);
            (cur_signs, next_signs) = (next_signs, cur_signs);
        }
        let last_cw = self.last_cw;
        let root_bit = self.root_bit;
        let output = cur_seeds.into_iter().zip(cur_signs).map(move |(s, t)| {
            let my_last_cw = Output::from(s);
            let o = if t { my_last_cw + last_cw } else { my_last_cw };
            if root_bit {
                o.neg()
            } else {
                o
            }
        });
        output
    }
    pub fn eval_all(&self) -> Vec<Output> {
        self.eval_all_into().collect()
    }
}
pub fn int_to_bits(mut v: usize, width: usize) -> Vec<bool> {
    let mut output = vec![false; width];
    for i in 0..width {
        let b = v & 1 == 1;
        v >>= 1;
        output[width - i - 1] = b;
    }
    output
}
#[cfg(test)]
mod tests {
    use rand::{thread_rng, RngCore};

    use crate::field::PrimeField64x2;

    use super::{DpfKey, Node};

    #[test]
    fn test_dpf() {
        const DEPTH: usize = 12;
        let mut rng = thread_rng();
        let root_0 = Node::random(&mut rng);
        let root_1 = Node::random(&mut rng);
        let roots = (root_0, root_1);
        let point = (rng.next_u64() & ((1 << DEPTH) - 1)) as u128;
        // let alpha_idx = [point << (128 - DEPTH)].into();
        let alpha_idx = point << (128 - DEPTH);
        let beta = PrimeField64x2::random(&mut rng);
        let (k_0, k_1) = DpfKey::gen(&roots, &alpha_idx, DEPTH, &beta);
        let eval_all_0 = k_0.eval_all();
        let eval_all_1 = k_1.eval_all();
        for i in 0usize..1 << DEPTH {
            let input = (i as u128) << (128 - DEPTH);
            let mut bs_output_0 = PrimeField64x2::default();
            let mut bs_output_1 = PrimeField64x2::default();
            k_0.eval(&input, &mut bs_output_0);
            k_1.eval(&input, &mut bs_output_1);
            assert_eq!(bs_output_0, eval_all_0[i]);
            assert_eq!(bs_output_1, eval_all_1[i]);
            let bs_output = bs_output_0 + bs_output_1;
            if (i as u128) != point {
                assert_eq!(bs_output, PrimeField64x2::default());
            }
            if (i as u128) == point {
                // core::array::from_fn(|i| u128::from(bs_output_0[i] ^ bs_output_1[i])).into();
                assert_eq!(bs_output, beta);
            }
        }
    }
}
