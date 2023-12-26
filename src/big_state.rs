use std::ops::BitXorAssign;
use std::time::Instant;

use aes_prng::AesRng;

use crate::dpf::convert_into;
use crate::dpf::tree_and_leaf_depth;
use crate::dpf::BitVec;
use crate::dpf::EvalAllResult;
use crate::dpf::ExpandedNode;
use crate::dpf::Node;
use crate::trie::BinaryTrie;

use super::BITS_OF_SECURITY;

#[derive(Clone, Debug)]
struct BigStateLastCorrectionWord {
    nodes: Box<[Node]>,
    nodes_per_output: usize,
}
impl BigStateLastCorrectionWord {
    fn new(output_bits: usize, point_count: usize) -> Self {
        let nodes_for_single_output = (output_bits + BITS_OF_SECURITY - 1) / BITS_OF_SECURITY;
        let total_nodes = nodes_for_single_output * point_count;
        Self {
            nodes: Box::from(vec![Node::default(); total_nodes]),
            nodes_per_output: nodes_for_single_output,
        }
    }
    fn set_output(&mut self, output_idx: usize, nodes: &[Node]) {
        let start = output_idx * self.nodes_per_output;
        for i in 0..self.nodes_per_output {
            self.nodes[start + i] = nodes[i];
        }
    }
    fn xor_output(&mut self, output_idx: usize, nodes: &[Node]) {
        let start = output_idx * self.nodes_per_output;
        for i in 0..self.nodes_per_output {
            self.nodes[start + i] ^= &nodes[i];
        }
    }
}
#[derive(Clone)]
struct BigStateCorrectionWord {
    nodes: Box<[Node]>,
}

impl BigStateCorrectionWord {
    pub fn single_node_count(t: usize) -> usize {
        1 + 2 * ((t + BITS_OF_SECURITY - 1) / BITS_OF_SECURITY)
    }
    pub fn get_entry(&self, index: usize, t: usize) -> &[Node] {
        let single_value_node_count = Self::single_node_count(t);
        &self.nodes[single_value_node_count * index..single_value_node_count * (index + 1)]
    }
    pub fn get_entry_mut(&mut self, index: usize, t: usize) -> &mut [Node] {
        let single_value_node_count = Self::single_node_count(t);
        &mut self.nodes[single_value_node_count * index..single_value_node_count * (index + 1)]
    }
    pub fn set_node(&mut self, node: &Node, index: usize, t: usize) {
        let entry = self.get_entry_mut(index, t);
        entry[0] = *node;
    }
    fn bit_coords(index: usize, direction: bool, t: usize) -> (usize, usize) {
        let single_count = Self::single_node_count(t);
        let bits_node_per_direction = (single_count - 1) / 2;
        let entry = 1 + (index / BITS_OF_SECURITY) + (direction as usize) * bits_node_per_direction;
        let bit_index = index & (BITS_OF_SECURITY - 1);
        (entry, bit_index)
    }
    pub fn set_bits_from(&mut self, entry_idx: usize, t: usize, bits: &[Node]) {
        let entry = self.get_entry_mut(entry_idx, t);
        entry[1..].copy_from_slice(bits);
    }
    pub fn xor_bits_from(&mut self, entry_idx: usize, t: usize, bits: &[Node]) {
        let entry = self.get_entry_mut(entry_idx, t);
        entry[1..]
            .iter_mut()
            .zip(bits.iter())
            .for_each(|(entry, bits)| {
                *entry ^= bits;
            })
    }
    pub fn toggle_bit(&mut self, entry_idx: usize, bit_index: usize, direction: bool, t: usize) {
        let entry = self.get_entry_mut(entry_idx, t);
        let (cell, bit_index) = Self::bit_coords(bit_index, direction, t);
        entry[cell].toggle_bit(bit_index);
    }
    pub fn get_bits(entry: &[Node], direction: bool, t: usize) -> &[Node] {
        let (first_cell, _) = Self::bit_coords(0, direction, t);
        let (last_cell, _) = Self::bit_coords(t - 1, direction, t);
        &entry[first_cell..=last_cell]
    }
    pub fn correct(&self, sign: &BitVec, output: &mut [Node]) {
        let t = sign.len();
        output.iter_mut().for_each(|node| node.zero());
        for i in 0..sign.len() {
            let bit = sign.get(i);
            let entry = self.get_entry(i, t);
            if bit {
                output
                    .iter_mut()
                    .zip(entry.iter())
                    .for_each(|(output, entry)| {
                        *output ^= entry;
                    });
            }
        }
    }
}
fn gen_cw(
    i: usize,
    alphas: &BinaryTrie,
    seed_0: &[Node],
    seed_1: &[Node],
    container_0: &mut ExpandedNode,
    container_1: &mut ExpandedNode,
) -> BigStateCorrectionWord {
    let t = seed_0.len();
    let single_size = BigStateCorrectionWord::single_node_count(t);
    let total_nodes = t * single_size;
    let mut rng = AesRng::from_random_seed();
    let mut cw = BigStateCorrectionWord {
        nodes: Box::from(
            (0..total_nodes)
                .map(|_| Node::random(&mut rng))
                .collect::<Vec<_>>(),
        ),
    };
    let iter = alphas.iter_at_depth(i);
    let mut next_idx: usize = 0;
    iter.enumerate().for_each(|(idx, node)| {
        container_0.fill_all(&seed_0[idx]);
        container_1.fill_all(&seed_1[idx]);
        let mut delta_left = *container_0.get_node(false);
        delta_left ^= container_1.get_node(false);
        let mut delta_right = *container_0.get_node(true);
        delta_right ^= container_1.get_node(true);
        let deltas = [delta_left, delta_right];
        let left_son = node.borrow().get_son(false);
        let right_son = node.borrow().get_son(true);
        let d = next_idx;
        cw.set_bits_from(idx, t, container_0.get_bits());
        cw.xor_bits_from(idx, t, container_1.get_bits());
        if left_son.is_some() && right_son.is_some() {
            next_idx += 2;
            let r = Node::random(&mut rng);
            cw.set_node(&r, idx, t);
            cw.toggle_bit(idx, d, false, t);
            cw.toggle_bit(idx, d + 1, true, t);
        } else {
            let z = right_son.is_some();
            next_idx += 1;
            cw.set_node(&deltas[(!z) as usize], idx, t);
            cw.toggle_bit(idx, d, z, t);
        }
    });
    cw
}
fn gen_conv_cw(
    alphas_betas: &[(BitVec, BitVec)],
    seed_0: &[Node],
    seed_1: &[Node],
) -> BigStateLastCorrectionWord {
    let t = alphas_betas.len();
    let input_bits = alphas_betas[0].0.len();
    let output_bits = alphas_betas[0].1.len();
    let (tree_depth, _) = tree_and_leaf_depth(input_bits, output_bits);
    let single_output_nodes = (output_bits + BITS_OF_SECURITY - 1) / BITS_OF_SECURITY;
    let mut nodes = vec![Node::default(); single_output_nodes];
    let mut cw = BigStateLastCorrectionWord::new(output_bits, t);
    for k in 0..t {
        convert_into(&seed_0[k], &mut nodes.as_mut());
        cw.set_output(k, nodes.as_ref());
        convert_into(&seed_1[k], &mut nodes.as_mut());
        cw.xor_output(k, nodes.as_ref());
        if single_output_nodes > 1 {
            cw.xor_output(k, alphas_betas[k].1.as_ref());
        } else {
            // The output is just a single node...
            nodes[0] = Node::default();
            let output_pos = {
                let mut pos = 0;
                for bit in tree_depth..input_bits {
                    pos <<= 1;
                    pos += alphas_betas[k].0.get(bit) as usize;
                }
                pos
            };
            let output_pos = output_pos * output_bits;
            for i in 0..output_bits {
                nodes[0].set_bit(output_pos + i, alphas_betas[k].1.get(i));
            }
            cw.xor_output(k, nodes.as_ref());
        }
    }
    cw
}
fn conv_correct_xor_into(sign: &BitVec, cw: &BigStateLastCorrectionWord, output: &mut [Node]) {
    cw.nodes
        .chunks_exact(cw.nodes_per_output)
        .enumerate()
        .for_each(|(idx, cwi)| {
            let bit = sign.get(idx);
            if bit {
                output.iter_mut().zip(cwi.iter()).for_each(|(out, inp)| {
                    out.bitxor_assign(inp);
                });
            }
        })
}

pub struct BigStateDpfKey {
    root: Node,
    root_sign: bool,
    cws: Vec<BigStateCorrectionWord>,
    last_cw: BigStateLastCorrectionWord,
    input_len: usize,
    output_len: usize,
}
impl BigStateDpfKey {
    pub fn gen(
        alphas_betas: &[(BitVec, BitVec)],
        roots: &(Node, Node),
    ) -> (BigStateDpfKey, BigStateDpfKey) {
        // We assume alphas_betas is SORTED!
        let mut alphas = BinaryTrie::default();
        for (alpha, _) in alphas_betas {
            alphas.insert(&alpha.into());
        }
        let t = alphas_betas.len();
        let input_bits = alphas_betas[0].0.len();
        let output_bits = alphas_betas[0].1.len();
        let mut seed_0_a = vec![Node::default(); t];
        let mut seed_1_a = vec![Node::default(); t];
        let mut seed_0_b = vec![Node::default(); t];
        let mut seed_1_b = vec![Node::default(); t];
        seed_0_a[0] = roots.0;
        seed_1_a[0] = roots.1;
        let mut sign_0_a = vec![BitVec::new(t); t];
        let mut sign_1_a = vec![BitVec::new(t); t];
        let mut sign_0_b = vec![BitVec::new(t); t];
        let mut sign_1_b = vec![BitVec::new(t); t];
        sign_1_a[0].set(0, true);
        let mut container_0 = ExpandedNode::new(t);
        let mut container_1 = ExpandedNode::new(t);
        let (tree_depth, _) = tree_and_leaf_depth(input_bits, output_bits);
        let mut correction_container_0 =
            vec![Node::default(); BigStateCorrectionWord::single_node_count(t)];
        let mut correction_container_1 =
            vec![Node::default(); BigStateCorrectionWord::single_node_count(t)];
        let mut cws = Vec::with_capacity(tree_depth);
        let mut seed_0_prev = &mut seed_0_a;
        let mut seed_0_cur = &mut seed_0_b;
        let mut seed_1_prev = &mut seed_1_a;
        let mut seed_1_cur = &mut seed_1_b;
        let mut sign_0_prev = &mut sign_0_a;
        let mut sign_0_cur = &mut sign_0_b;
        let mut sign_1_prev = &mut sign_1_a;
        let mut sign_1_cur = &mut sign_1_b;
        for i in 0..tree_depth {
            let cw = gen_cw(
                i,
                &alphas,
                &seed_0_prev,
                &seed_1_prev,
                &mut container_0,
                &mut container_1,
            );
            let mut state_idx = 0;
            let time = Instant::now();
            alphas.iter_at_depth(i).enumerate().count();
            println!("Iter at depth: {} took: {}", i, time.elapsed().as_micros());
            for (k, node) in alphas.iter_at_depth(i).enumerate() {
                cw.correct(&sign_0_prev[k], &mut correction_container_0);
                cw.correct(&sign_1_prev[k], &mut correction_container_1);
                container_0.fill_all(&seed_0_prev[k]);
                container_1.fill_all(&seed_1_prev[k]);
                for z in [false, true] {
                    if node.borrow().get_son(z).is_some() {
                        // Handle seed.
                        // First key
                        let node = container_0.get_node(z);
                        seed_0_cur[state_idx] = correction_container_0[0];
                        seed_0_cur[state_idx].bitxor_assign(node);
                        // Second key
                        let node = container_1.get_node(z);
                        seed_1_cur[state_idx] = correction_container_1[0];
                        seed_1_cur[state_idx].bitxor_assign(node);

                        // Handle sign.
                        let expanded_bits_0 = container_0.get_bits_direction(z);
                        let expanded_bits_1 = container_1.get_bits_direction(z);
                        let correction_bits_0 =
                            BigStateCorrectionWord::get_bits(&correction_container_0, z, t);
                        let correction_bits_1 =
                            BigStateCorrectionWord::get_bits(&correction_container_1, z, t);
                        sign_0_cur[state_idx]
                            .as_mut()
                            .iter_mut()
                            .zip(expanded_bits_0.iter())
                            .zip(correction_bits_0.iter())
                            .for_each(|((out, in1), in2)| {
                                *out = *in1;
                                out.bitxor_assign(in2);
                            });
                        sign_1_cur[state_idx]
                            .as_mut()
                            .iter_mut()
                            .zip(expanded_bits_1.iter())
                            .zip(correction_bits_1.iter())
                            .for_each(|((out, in1), in2)| {
                                *out = *in1;
                                out.bitxor_assign(in2);
                            });
                        state_idx += 1;
                    }
                }
            }
            cws.push(cw);
            (sign_0_prev, sign_0_cur) = (sign_0_cur, sign_0_prev);
            (sign_1_prev, sign_1_cur) = (sign_1_cur, sign_1_prev);
            (seed_0_prev, seed_0_cur) = (seed_0_cur, seed_0_prev);
            (seed_1_prev, seed_1_cur) = (seed_1_cur, seed_1_prev);
        }
        // Handle last CW.
        let last_cw = gen_conv_cw(alphas_betas, &seed_0_prev, &seed_1_prev);
        let first_key = BigStateDpfKey {
            root: roots.0,
            root_sign: false,
            cws: cws.clone(),
            last_cw: last_cw.clone(),
            input_len: input_bits,
            output_len: output_bits,
        };
        let second_key = BigStateDpfKey {
            root: roots.1,
            root_sign: true,
            cws,
            last_cw,
            input_len: input_bits,
            output_len: output_bits,
        };
        (first_key, second_key)
    }
    fn non_zero_point_count(&self) -> usize {
        self.last_cw.nodes.len() / self.last_cw.nodes_per_output
    }
    pub fn make_aux_variables(&self) -> (BitVec, Box<[Node]>, ExpandedNode) {
        let t = self.non_zero_point_count();
        let sign_contianer = BitVec::new(t);
        let correction_container =
            vec![Node::default(); BigStateCorrectionWord::single_node_count(t)];
        let expanded_node = ExpandedNode::new(t);
        (sign_contianer, correction_container.into(), expanded_node)
    }
    pub fn eval(
        &self,
        x: &BitVec,
        y: &mut BitVec,
        sign_container: &mut BitVec,
        correction_container: &mut Box<[Node]>,
        expansion_container: &mut ExpandedNode,
    ) {
        sign_container.zero();
        debug_assert_eq!(x.len(), self.input_len);
        debug_assert_eq!(y.len(), self.output_len);
        let t = self.non_zero_point_count();
        sign_container.zero();
        let (tree_depth, _) = tree_and_leaf_depth(self.input_len, self.output_len);
        let mut seed = self.root;
        sign_container.set(0, self.root_sign);
        for i in 0..tree_depth {
            self.cws[i].correct(sign_container, &mut correction_container[..]);
            let path_bit = x.get(i);
            expansion_container.fill(&seed, path_bit);
            seed = *expansion_container.get_node(path_bit);
            seed.bitxor_assign(&correction_container[0]);
            sign_container
                .as_mut()
                .iter_mut()
                .zip(expansion_container.get_bits_direction(path_bit).iter())
                .zip(BigStateCorrectionWord::get_bits(
                    &correction_container,
                    path_bit,
                    t,
                ))
                .for_each(|((output, in1), in2)| {
                    *output = *in1;
                    output.bitxor_assign(in2);
                });
        }
        convert_into(&seed, y.as_mut());
        conv_correct_xor_into(&sign_container, &self.last_cw, y.as_mut());
        if x.len() > tree_depth {
            let pos_in_tree = {
                let mut pos = 0;
                for idx in tree_depth..x.len() {
                    pos <<= 1;
                    pos ^= x.get(idx) as usize;
                }
                pos
            };
            let idx_start = pos_in_tree * self.output_len;
            y.as_mut()[0].shr((BITS_OF_SECURITY - idx_start - self.output_len) as u32);
            y.as_mut()[0].shl((BITS_OF_SECURITY - self.output_len) as u32);
        }
        y.normalize();
    }
    pub fn eval_all(&self) -> EvalAllResult {
        let t = self.non_zero_point_count();
        let nodes_per_item = (self.output_len + BITS_OF_SECURITY - 1) / BITS_OF_SECURITY;
        let (tree_depth, leaf_depth) = tree_and_leaf_depth(self.input_len, self.output_len);
        let total_nodes = (1 << tree_depth) * nodes_per_item;
        let mut output = Vec::with_capacity(total_nodes);
        unsafe { output.set_len(total_nodes) };
        let mut directions = Vec::with_capacity(tree_depth);
        let (mut sign_container_static, mut correction_container, mut expansion_container) =
            self.make_aux_variables();
        sign_container_static.set(0, self.root_sign);
        let mut sign_container = &mut sign_container_static;
        let mut tmp_storage = vec![sign_container.clone(); tree_depth];
        let mut seed = self.root;
        let mut output_iter = output.chunks_mut(nodes_per_item);
        'outer: loop {
            let depth = directions.len();
            if depth == tree_depth {
                let output_container = output_iter.next().unwrap();
                convert_into(&seed, output_container);
                conv_correct_xor_into(&sign_container, &self.last_cw, output_container);
                loop {
                    let (dir, s_r): (bool, Node) = match directions.pop() {
                        None => break 'outer,
                        Some(tuple) => tuple,
                    };
                    if !dir {
                        // We finished left, go to right
                        sign_container = &mut tmp_storage[directions.len()];
                        seed = s_r;
                        directions.push((true, seed));
                        break;
                    }
                    // Otherwise we finished right, keep going up
                }
            } else {
                expansion_container.fill_all(&seed);
                self.cws[depth].correct(sign_container, &mut correction_container[..]);
                seed = *expansion_container.get_node(false);
                seed.bitxor_assign(&correction_container[0]);
                let mut seed_right = *expansion_container.get_node(true);
                seed_right.bitxor_assign(&correction_container[0]);
                sign_container_static
                    .as_mut()
                    .iter_mut()
                    .zip(expansion_container.get_bits_direction(false).iter())
                    .zip(BigStateCorrectionWord::get_bits(
                        &correction_container,
                        false,
                        t,
                    ))
                    .for_each(|((output, in1), in2)| {
                        *output = *in1;
                        output.bitxor_assign(in2);
                    });
                tmp_storage[depth]
                    .as_mut()
                    .iter_mut()
                    .zip(expansion_container.get_bits_direction(true).iter())
                    .zip(BigStateCorrectionWord::get_bits(
                        &correction_container,
                        true,
                        t,
                    ))
                    .for_each(|((output, in1), in2)| {
                        *output = *in1;
                        output.bitxor_assign(in2);
                    });
                directions.push((false, seed_right));
                sign_container = &mut sign_container_static;
            }
        }
        assert!(output_iter.next().is_none());
        EvalAllResult::new(output, self.output_len, leaf_depth)
    }
}

#[cfg(test)]
mod test {
    use std::{collections::HashMap, ops::BitXorAssign};

    use aes_prng::AesRng;
    use rand::{thread_rng, RngCore};

    use crate::{
        big_state::BigStateDpfKey,
        dpf::{int_to_bits, BitVec, Node},
        BITS_OF_SECURITY,
    };

    #[test]
    fn test_dpf_single_point() {
        const DEPTH: usize = 1;
        const OUTPUT_WIDTH: usize = 65;
        const POINTS: usize = 2;
        let mut rng = AesRng::from_random_seed();
        let root_0 = Node::default();
        let root_1 = Node::default();
        let roots = (root_0, root_1);
        let mut alphas_betas = Vec::new();
        let mut ab_map = HashMap::new();
        while ab_map.len() < POINTS {
            let alpha_idx = (rng.next_u32() as usize) & ((1 << DEPTH) - 1);
            if ab_map.contains_key(&alpha_idx) {
                continue;
            }
            let alpha: Vec<_> = int_to_bits(alpha_idx, DEPTH);
            let alpha_v = BitVec::from(&alpha[..]);
            let beta: Vec<bool> = (0..OUTPUT_WIDTH).map(|_| rng.next_u32() & 1 == 1).collect();
            let beta_bitvec = BitVec::from(&beta[..]);
            ab_map.insert(alpha_idx, beta_bitvec.clone());
            alphas_betas.push((alpha_v, beta_bitvec));
        }
        alphas_betas.sort();
        let (k_0, k_1) = BigStateDpfKey::gen(&alphas_betas, &roots);
        let mut output_0 = BitVec::new(OUTPUT_WIDTH);
        let mut output_1 = BitVec::new(OUTPUT_WIDTH);
        let (mut signs, mut corrections, mut expanded_node) = k_0.make_aux_variables();
        for i in 0usize..1 << DEPTH {
            let bits_i = int_to_bits(i, DEPTH);
            let input = BitVec::from(&bits_i[..]);
            let mut bs_output_0 = &mut output_0;
            let mut bs_output_1 = &mut output_1;
            k_0.eval(
                &input,
                &mut bs_output_0,
                &mut signs,
                &mut corrections,
                &mut expanded_node,
            );
            k_1.eval(
                &input,
                &mut bs_output_1,
                &mut signs,
                &mut corrections,
                &mut expanded_node,
            );
            if !ab_map.contains_key(&i) {
                assert_eq!(&bs_output_0, &bs_output_1);
            } else {
                let beta_bitvec = ab_map.get(&i).unwrap();
                output_0.bitxor_assign(&output_1);
                assert_eq!(&output_0, beta_bitvec);
            }
        }
    }
    #[test]
    fn test_dpf_evalall() {
        const DEPTH: usize = 1;
        const OUTPUT_WIDTH: usize = 65;
        const POINTS: usize = 2;
        let mut rng = thread_rng();
        let root_0 = Node::random(&mut rng);
        let root_1 = Node::random(&mut rng);
        let roots = (root_0, root_1);
        let mut alphas_betas = Vec::new();
        let mut ab_map = HashMap::new();
        while ab_map.len() < POINTS {
            let alpha_idx = (rng.next_u32() as usize) & ((1 << DEPTH) - 1);
            if ab_map.contains_key(&alpha_idx) {
                continue;
            }
            let alpha: Vec<_> = int_to_bits(alpha_idx, DEPTH);
            let alpha_v = BitVec::from(&alpha[..]);
            let beta: Vec<bool> = (0..OUTPUT_WIDTH).map(|_| rng.next_u32() & 1 == 1).collect();
            let beta_bitvec = BitVec::from(&beta[..]);
            ab_map.insert(alpha_idx, beta_bitvec.clone());
            alphas_betas.push((alpha_v, beta_bitvec));
        }
        alphas_betas.sort();
        let (k_0, k_1) = BigStateDpfKey::gen(&alphas_betas, &roots);
        let ev_all_0 = k_0.eval_all();
        let ev_all_1 = k_1.eval_all();
        let blocks_per_output = (OUTPUT_WIDTH + BITS_OF_SECURITY - 1) / BITS_OF_SECURITY;
        let mut output_0 = vec![Node::default(); blocks_per_output];
        let mut output_1 = vec![Node::default(); blocks_per_output];
        for i in 0usize..1 << DEPTH {
            ev_all_0.get_item(i, &mut output_0);
            ev_all_1.get_item(i, &mut output_1);
            if !ab_map.contains_key(&i) {
                assert_eq!(output_0, output_1);
            } else {
                let beta_bitvec = ab_map.get(&i).unwrap();
                output_0
                    .iter_mut()
                    .zip(output_1.iter())
                    .for_each(|(output, input)| output.bitxor_assign(input));
                assert_eq!(output_0, beta_bitvec.as_ref());
            }
        }
    }
}
