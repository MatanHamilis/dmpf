use std::ops::BitXorAssign;

use aes_prng::AesRng;

use crate::dpf::convert_into;
use crate::dpf::many_prg;
use crate::dpf::tree_and_leaf_depth;
use crate::dpf::BitSliceMut;
use crate::dpf::BitVec;
use crate::dpf::ExpandedNode;
use crate::dpf::Node;
use crate::trie::BinaryTrie;

use super::BITS_OF_SECURITY;

#[derive(Clone)]
pub struct BigStateLastCorrectionWord {
    nodes: Box<[Node]>,
    nodes_per_output: usize,
}
impl BigStateLastCorrectionWord {
    fn new(output_bits: usize, point_count: usize) -> Self {
        let nodes_for_single_output = (output_bits + BITS_OF_SECURITY - 1) / output_bits;
        let total_nodes = nodes_for_single_output * point_count;
        Self {
            nodes: Box::from(vec![Node::default(); total_nodes]),
            nodes_per_output: nodes_for_single_output,
        }
    }
    fn set_output(&mut self, output_idx: usize, nodes: &[Node]) {
        for i in 0..self.nodes_per_output {
            self.nodes[i] = nodes[i];
        }
    }
    fn xor_output(&mut self, output_idx: usize, nodes: &[Node]) {
        for i in 0..self.nodes_per_output {
            self.nodes[i] ^= &nodes[i];
        }
    }
}
#[derive(Clone)]
pub struct BigStateCorrectionWord {
    nodes: Box<[Node]>,
}

impl BigStateCorrectionWord {
    pub fn new(t: usize) -> Self {
        let node_count = t * Self::single_node_count(t);
        Self {
            nodes: Box::from(vec![Node::default(); node_count]),
        }
    }
    pub fn single_node_count(t: usize) -> usize {
        1 + 2 * ((t + BITS_OF_SECURITY - 1) / BITS_OF_SECURITY)
    }
    pub fn get_entry(&self, index: usize, t: usize) -> &[Node] {
        let single_value_node_count = Self::single_node_count(t);
        &self.nodes[t * index..t * (index + 1)]
    }
    pub fn get_entry_mut(&mut self, index: usize, t: usize) -> &mut [Node] {
        let single_value_node_count = Self::single_node_count(t);
        &mut self.nodes[t * index..t * (index + 1)]
    }
    pub fn set_node(&mut self, node: &Node, index: usize, t: usize) {
        let entry = self.get_entry_mut(index, t);
        entry[0] = *node;
    }
    fn bit_coords(index: usize, direction: bool, t: usize) -> (usize, usize) {
        let single_count = Self::single_node_count(t);
        let index = (direction as usize) * single_count + index;
        let bit_index = index & (BITS_OF_SECURITY - 1);
        let entry = index / BITS_OF_SECURITY + (direction as usize) * single_count;
        (entry, bit_index)
    }
    pub fn set_bit(
        &mut self,
        bit_value: bool,
        entry_idx: usize,
        bit_index: usize,
        direction: bool,
        t: usize,
    ) {
        let entry = self.get_entry_mut(entry_idx, t);
        let (cell, bit_index) = Self::bit_coords(bit_index, direction, t);
        entry[cell].set_bit(bit_index, bit_value)
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
    pub fn get_bit(
        &mut self,
        entry_idx: usize,
        bit_index: usize,
        direction: bool,
        t: usize,
    ) -> bool {
        let entry = self.get_entry_mut(entry_idx, t);
        let (cell, bit_index) = Self::bit_coords(bit_index, direction, t);
        entry[cell].get_bit(bit_index)
    }
    pub fn get_bits(entry: &[Node], direction: bool, t: usize) -> &[Node] {
        let (first_cell, _) = Self::bit_coords(0, direction, t);
        let (last_cell, _) = Self::bit_coords(t - 1, direction, t);
        &entry[first_cell..=last_cell]
    }
    pub fn correct(&self, sign: &BitVec, output: &mut [Node]) {
        let t = sign.len();
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
    let mut iter = alphas.iter_at_depth(i);
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
        for i in 0..t {
            cw.set_bits_from(idx, t, container_0.get_bits());
            cw.xor_bits_from(idx, t, container_1.get_bits());
        }
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
    let (tree_depth, leaf_depth) = tree_and_leaf_depth(input_bits, output_bits);
    let single_output_nodes = (output_bits + BITS_OF_SECURITY - 1) / BITS_OF_SECURITY;
    let total_nodes = single_output_nodes * alphas_betas.len();
    let mut nodes = vec![Node::default(); single_output_nodes];
    let mut cw = BigStateLastCorrectionWord::new(output_bits, t);
    for k in 0..t {
        many_prg(&seed_0[k], 0..single_output_nodes as u16, &mut nodes);
        cw.set_output(k, &nodes);
        many_prg(&seed_1[k], 0..single_output_nodes as u16, &mut nodes);
        cw.xor_output(k, &nodes);
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
            cw.xor_output(k, &nodes);
        }
    }
    cw
}
fn conv_correct_xor_into(sign: &BitVec, cw: &BigStateLastCorrectionWord, output: &[Node]) {
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

struct BigStateDpfKey {
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
        roots: (Node, Node),
    ) -> (BigStateDpfKey, BigStateDpfKey) {
        // We assume alphas_betas is SORTED!
        let mut alphas = BinaryTrie::default();
        for (alpha, _) in alphas_betas {
            alphas.insert(alpha);
        }
        let t = alphas_betas.len();
        let input_bits = alphas_betas[0].0.len();
        let output_bits = alphas_betas[0].1.len();
        let mut seed_0 = vec![Node::default(); t];
        let mut seed_1 = vec![Node::default(); t];
        seed_0[0] = roots.0;
        seed_1[0] = roots.1;
        let mut sign_0 = vec![BitVec::new(t); t];
        let mut sign_1 = vec![BitVec::new(t); t];
        sign_1[0].set(0, true);
        let mut container_0 = ExpandedNode::new(t);
        let mut container_1 = ExpandedNode::new(t);
        let (tree_depth, leaf_depth) = tree_and_leaf_depth(input_bits, output_bits);
        let mut correction_container_0 =
            vec![Node::default(); BigStateCorrectionWord::single_node_count(t)];
        let mut correction_container_1 =
            vec![Node::default(); BigStateCorrectionWord::single_node_count(t)];
        let mut cws = Vec::with_capacity(tree_depth);
        for i in 0..tree_depth {
            let cw = gen_cw(
                i,
                &alphas,
                &seed_0,
                &seed_1,
                &mut container_0,
                &mut container_1,
            );
            let mut state_idx = 0;
            for (k, node) in alphas.iter_at_depth(i).enumerate() {
                cw.correct(&sign_0[k], &mut correction_container_0);
                cw.correct(&sign_1[k], &mut correction_container_1);
                container_0.fill_all(&seed_0[k]);
                container_1.fill_all(&seed_1[k]);
                for z in [false, true] {
                    if node.borrow().get_son(z).is_some() {
                        // Handle seed.
                        // First key
                        let node = container_0.get_node(z);
                        correction_container_0[0].bitxor_assign(node);
                        seed_0[state_idx] = correction_container_0[0];
                        // Second key
                        let node = container_1.get_node(z);
                        correction_container_1[0].bitxor_assign(node);
                        seed_1[state_idx] = correction_container_1[0];

                        // Handle sign.
                        let expanded_bits_0 = container_0.get_bits_direction(z);
                        let expanded_bits_1 = container_1.get_bits_direction(z);
                        let correction_bits_0 =
                            BigStateCorrectionWord::get_bits(&correction_container_0, z, t);
                        let correction_bits_1 =
                            BigStateCorrectionWord::get_bits(&correction_container_1, z, t);
                        sign_0[state_idx]
                            .as_mut()
                            .iter_mut()
                            .zip(expanded_bits_0.iter())
                            .zip(correction_bits_0.iter())
                            .for_each(|((out, in1), in2)| {
                                out.bitxor_assign(in1);
                                out.bitxor_assign(in2);
                            });
                        sign_1[state_idx]
                            .as_mut()
                            .iter_mut()
                            .zip(expanded_bits_1.iter())
                            .zip(correction_bits_1.iter())
                            .for_each(|((out, in1), in2)| {
                                out.bitxor_assign(in1);
                                out.bitxor_assign(in2);
                            });
                        state_idx += 1;
                    }
                }
            }
            cws.push(cw);
        }
        // Handle last CW.
        let last_cw = gen_conv_cw(alphas_betas, &seed_0, &seed_1);
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
        if self.output_len >= BITS_OF_SECURITY {
            //easy
            let mut bit_slice = BitSliceMut::from(y);
            convert_into(&seed, &mut bit_slice);
            conv_correct_xor_into(&sign_container, &self.last_cw, y.as_mut());
        }
    }
}
