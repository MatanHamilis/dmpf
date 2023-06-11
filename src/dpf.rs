use once_cell::sync::Lazy;
use std::ops::AddAssign;
use std::ops::BitXor;
use std::ops::BitXorAssign;

use crate::xor_arrays;
use crate::BITS_IN_BYTE;

use super::BITS_OF_SECURITY;
use super::BYTES_OF_SECURITY;
use aes::{
    cipher::{BlockEncrypt, KeyInit},
    Aes128, Block,
};
use rand::SeedableRng;
use rand::{CryptoRng, RngCore};
const DPF_AES_KEY: [u8; BYTES_OF_SECURITY] =
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2];
static AES: Lazy<Aes128> = Lazy::new(|| Aes128::new_from_slice(&DPF_AES_KEY).unwrap());

fn double_prg(input: &Node) -> [Node; 2] {
    let mut blocks = [Block::from(*input.as_ref()); 2];
    blocks[1][0] ^= 1;
    AES.encrypt_blocks(&mut blocks);
    xor_arrays(&mut blocks[0].into(), input.as_ref());
    xor_arrays(&mut blocks[1].into(), input.as_ref());
    blocks[1][0] ^= 1;
    unsafe { std::mem::transmute(blocks) }
}
fn triple_prg(input: &Node) -> [Node; 3] {
    let mut blocks = [Block::from(*input.as_ref()); 3];
    blocks[1][0] ^= 1;
    blocks[2][0] ^= 2;
    AES.encrypt_blocks(&mut blocks);
    xor_arrays(&mut blocks[0].into(), input.as_ref());
    xor_arrays(&mut blocks[1].into(), input.as_ref());
    xor_arrays(&mut blocks[2].into(), input.as_ref());
    blocks[1][0] ^= 1;
    blocks[2][0] ^= 2;
    unsafe { std::mem::transmute(blocks) }
}
fn many_prg(input: &Node, output: &mut [Node]) {
    let input_block = Block::from(*input.as_ref());
    if output.len() > 255 {
        unimplemented!()
    }
    let mut blocks_output =
        unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut Block, output.len()) };
    blocks_output.iter_mut().enumerate().for_each(|(i, mut v)| {
        *v = input_block;
        v[0] ^= i as u8;
    });
    AES.encrypt_blocks(&mut blocks_output);
    blocks_output.iter_mut().enumerate().for_each(|(i, mut v)| {
        v[0] ^= i as u8;
    });
}
fn many_many_prg(input: &[Node], factor: usize, output: &mut [Node]) {
    assert!(input.len() * factor <= output.len());
    let input_block =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const Block, input.len()) };
    if factor > 255 {
        unimplemented!()
    }
    let mut output_block =
        unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut Block, output.len()) };
    output_block
        .chunks_exact_mut(factor)
        .zip(input_block.iter())
        .for_each(|(output_chunk, input)| {
            output_chunk.iter_mut().enumerate().for_each(|(idx, v)| {
                *v = *input;
                v[0] ^= idx as u8;
            })
        });
    AES.encrypt_blocks(&mut output_block);
    output_block
        .chunks_exact_mut(factor)
        .for_each(|output_chunk| {
            output_chunk.iter_mut().enumerate().for_each(|(idx, v)| {
                v[0] ^= idx as u8;
            })
        });
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Node([u64; BYTES_OF_SECURITY / 8]);
impl Node {
    pub fn random<R: CryptoRng + RngCore>(rng: &mut R) -> Self {
        let mut output = Node::default();
        output.fill(rng);
        output
    }
    pub fn fill(&mut self, mut rng: impl RngCore + CryptoRng) {
        let ptr = self.as_mut();
        rng.fill_bytes(ptr);
    }
    fn coordinates(index: usize) -> (usize, usize) {
        (index / u64::BITS as usize, index & (u64::BITS as usize - 1))
    }
    pub fn get_bit(&self, index: usize) -> bool {
        let (cell, bit) = Self::coordinates(index);
        self.0[cell] >> bit & 1 == 1
    }
    pub fn set_bit(&mut self, index: usize, value: bool) {
        let (cell, bit) = Self::coordinates(index);
        let mask: u64 = !(1 << bit);
        self.0[cell] = (self.0[cell] & mask) ^ ((value as u64) << bit)
    }
    pub fn mask(&mut self, bits: usize) {
        let index_to_mask = bits / (u64::BITS as usize);
        if (index_to_mask >= self.0.len()) {
            return;
        }
        let (mask, _) = u64::overflowing_shl(1, (bits & 63) as u32);
        let mask = mask.wrapping_sub(1);
        self.0[index_to_mask] &= mask;
    }
}
impl Default for Node {
    fn default() -> Self {
        Self([0u64; BYTES_OF_SECURITY / 8])
    }
}
impl BitXorAssign<&Node> for Node {
    fn bitxor_assign(&mut self, rhs: &Node) {
        for i in 0..self.0.len() {
            self.0[i] ^= rhs.0[i];
        }
    }
}
impl AsRef<[u8; 16]> for Node {
    fn as_ref(&self) -> &[u8; 16] {
        unsafe { (self.0.as_ptr() as *const [u8; 16]).as_ref().unwrap() }
    }
}
impl AsMut<[u8]> for Node {
    fn as_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.0.as_mut_ptr() as *mut u8, BYTES_OF_SECURITY) }
    }
}
struct ExpandedNode {
    node_l: Node,
    node_r: Node,
    bits: u8,
}
impl ExpandedNode {
    pub fn new() -> Self {
        ExpandedNode {
            node_l: Node::default(),
            node_r: Node::default(),
            bits: 0u8,
        }
    }
    pub fn fill(&mut self, node: &Node) {
        let expanded = triple_prg(node);
        self.node_l = expanded[0];
        self.node_r = expanded[1];
        self.bits = expanded[2].as_ref()[0];
    }
    pub fn get_left_bit(&self) -> bool {
        self.bits & 1 != 0
    }
    pub fn get_right_bit(&self) -> bool {
        self.bits & 2 != 0
    }
}
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BitVec {
    v: Box<[Node]>,
    len: usize,
}

impl BitVec {
    pub fn new(len: usize) -> Self {
        let nodes = (len + BITS_OF_SECURITY - 1) / BITS_OF_SECURITY;
        let v = vec![Node::default(); nodes].into();
        BitVec { v, len }
    }
    fn len(&self) -> usize {
        self.len
    }
    fn coordinates(index: usize) -> (usize, usize) {
        (index / BITS_OF_SECURITY, index & (BITS_OF_SECURITY - 1))
    }
    fn index(v: &u8, index: usize) -> bool {
        ((v >> index) & 1) == 1
    }
    fn get(&self, index: usize) -> bool {
        let (cell, idx) = Self::coordinates(index);
        self.v[cell].get_bit(idx)
    }
    fn set(&mut self, index: usize, val: bool) {
        let (cell, idx) = Self::coordinates(index);
        self.v[cell].set_bit(idx, val);
    }
    fn normalize(&mut self) {
        let bits_to_leave = self.len & (BITS_OF_SECURITY - 1);
        let bits_to_leave = ((bits_to_leave - 1) % BITS_OF_SECURITY) + 1;
        self.v.last_mut().unwrap().mask(bits_to_leave);
    }
}
#[derive(Debug, PartialEq, Eq)]
pub struct BitSliceMut<'a> {
    v: &'a mut [Node],
    len: usize,
}
impl<'a> AsMut<[Node]> for BitSliceMut<'a> {
    fn as_mut(&mut self) -> &mut [Node] {
        &mut self.v
    }
}
impl<'a> BitSliceMut<'a> {
    pub fn new(len: usize, v: &'a mut [Node]) -> Self {
        BitSliceMut { v, len }
    }
    fn len(&self) -> usize {
        self.len
    }
    fn coordinates(index: usize) -> (usize, usize) {
        (index / BITS_OF_SECURITY, index & (BITS_OF_SECURITY - 1))
    }
    fn get(&self, index: usize) -> bool {
        let (cell, idx) = Self::coordinates(index);
        self.v[cell].get_bit(idx)
    }
    fn set(&mut self, index: usize, val: bool) {
        let (cell, idx) = Self::coordinates(index);
        self.v[cell].set_bit(idx, val);
    }
    fn fill(&mut self, nodes: &[Node]) {
        self.v.copy_from_slice(nodes);
        self.normalize();
    }
    fn normalize(&mut self) {
        let bits_to_leave = self.len & (BITS_OF_SECURITY - 1);
        let bits_to_leave = ((bits_to_leave + BITS_OF_SECURITY - 1) % BITS_OF_SECURITY) + 1;
        self.v.last_mut().unwrap().mask(bits_to_leave);
    }
}
impl<'a> From<&'a mut BitVec> for BitSliceMut<'a> {
    fn from(value: &'a mut BitVec) -> Self {
        Self {
            v: &mut value.v,
            len: value.len,
        }
    }
}

impl From<&[bool]> for BitVec {
    fn from(v: &[bool]) -> Self {
        let bit_len = v.len();
        let len = (v.len() + BITS_OF_SECURITY - 1) / BITS_OF_SECURITY;
        let mut output = Vec::with_capacity(len);
        for chunk in v.chunks(BITS_OF_SECURITY) {
            let mut node = Node::default();
            for (idx, &bit) in chunk.iter().enumerate() {
                node.set_bit(idx, bit);
            }
            output.push(node);
        }
        Self {
            v: output.into(),
            len: bit_len,
        }
    }
}
impl From<(&[Node], usize)> for BitVec {
    fn from(value: (&[Node], usize)) -> Self {
        let v = value.0.to_vec();
        Self {
            v: v.into(),
            len: value.1,
        }
    }
}

impl BitXorAssign<&BitVec> for BitVec {
    fn bitxor_assign(&mut self, rhs: &BitVec) {
        for i in 0..self.v.len() {
            self.v[i].bitxor_assign(&rhs.v[i]);
        }
    }
}
impl<'a> BitXorAssign<&BitVec> for BitSliceMut<'a> {
    fn bitxor_assign(&mut self, rhs: &BitVec) {
        let bytes = (self.len + BITS_OF_SECURITY - 1) / BITS_OF_SECURITY;
        for i in 0..bytes {
            self.v[i].bitxor_assign(&rhs.v[i]);
        }
    }
}
#[derive(Clone, Copy)]
pub struct CorrectionWord {
    node: Node,
    bit_l: bool,
    bit_r: bool,
}
pub struct DpfKey {
    root: Node,
    root_bit: bool,
    cws: Vec<CorrectionWord>,
    last_cw: BitVec,
    input_bits: usize,
    output_bits: usize,
}
fn tree_and_leaf_depth(alpha_len: usize, beta_len: usize) -> (usize, usize) {
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

fn convert(node: &Node, bits: usize) -> BitVec {
    let mut output = BitVec::new(bits);
    convert_into(node, &mut BitSliceMut::from(&mut output));
    output
}
fn convert_into(node: &Node, output: &mut BitSliceMut) {
    let bits = output.len;
    if bits > BITS_OF_SECURITY {
        // We should expand node
        let nodes_num = (output.len + BITS_OF_SECURITY - 1) / BITS_OF_SECURITY;
        many_prg(node, output.as_mut());
    } else {
        // We don't have to expand node
        output.fill(std::slice::from_ref(node));
    }
}
impl DpfKey {
    pub fn gen(roots: (Node, Node), alpha: BitVec, beta: BitVec) -> (DpfKey, DpfKey) {
        let nodes_num = (beta.len() + BITS_OF_SECURITY - 1) / BITS_OF_SECURITY;
        let mut nodes = vec![Node::default(); nodes_num];
        let mut t_0 = false;
        let mut t_1 = true;
        let mut expanded_node_0 = ExpandedNode::new();
        let mut expanded_node_1 = ExpandedNode::new();
        let mut seed_0 = roots.0;
        let mut seed_1 = roots.1;
        let (tree_depth, leaf_depth) = tree_and_leaf_depth(alpha.len(), beta.len());
        let mut cws = Vec::with_capacity(tree_depth);
        for i in 0..tree_depth {
            expanded_node_0.fill(&seed_0);
            expanded_node_1.fill(&seed_1);
            let path_bit = alpha.get(i);
            let t_l_0 = expanded_node_0.get_left_bit();
            let t_l_1 = expanded_node_1.get_left_bit();
            let t_r_0 = expanded_node_0.get_right_bit();
            let t_r_1 = expanded_node_1.get_right_bit();
            let bit_left = !(t_l_0 ^ t_l_1 ^ path_bit);
            let bit_right = t_r_0 ^ t_r_1 ^ path_bit;
            let cw_node = if path_bit {
                // We go to the right, fix left sons.
                expanded_node_0.node_l ^= &expanded_node_1.node_l;
                seed_0 = expanded_node_0.node_r;
                seed_1 = expanded_node_1.node_r;
                expanded_node_0.node_l
            } else {
                // We go to the left, fix right sons.
                expanded_node_0.node_r ^= &expanded_node_1.node_r;
                seed_0 = expanded_node_0.node_l;
                seed_1 = expanded_node_1.node_l;
                expanded_node_0.node_r
            };
            if t_0 {
                seed_0 ^= &cw_node;
            }
            if t_1 {
                seed_1 ^= &cw_node;
            }
            (t_0, t_1) = if path_bit {
                (t_r_0 ^ (t_0 & bit_right), t_r_1 ^ (t_1 & bit_right))
            } else {
                (t_l_0 ^ (t_0 & bit_left), t_l_1 ^ (t_1 & bit_left))
            };
            cws.push(CorrectionWord {
                node: cw_node,
                bit_l: bit_left,
                bit_r: bit_right,
            });
        }
        let mut last_cw = BitVec::new(beta.len() << leaf_depth);
        let pos_in_subtree = {
            let mut pos = 0;
            for i in tree_depth..alpha.len() {
                pos <<= 1;
                if alpha.get(i) {
                    pos ^= 1;
                }
            }
            pos
        };
        let start = pos_in_subtree * beta.len();
        for j in 0..beta.len() {
            let bit = beta.get(j);
            last_cw.set(start + j, bit);
        }
        let mut conv = convert(&seed_0, beta.len() << leaf_depth);
        last_cw.bitxor_assign(&conv);
        convert_into(&seed_1, &mut BitSliceMut::from(&mut conv));
        last_cw.bitxor_assign(&conv);
        let first_key = DpfKey {
            root: roots.0,
            root_bit: false,
            cws: cws.clone(),
            last_cw: last_cw.clone(),
            input_bits: alpha.len(),
            output_bits: beta.len(),
        };
        let second_key = DpfKey {
            root: roots.1,
            root_bit: true,
            cws,
            last_cw,
            input_bits: alpha.len(),
            output_bits: beta.len(),
        };
        (first_key, second_key)
    }
    pub fn eval(&self, x: &BitVec, output: &mut BitSliceMut) {
        assert_eq!(x.len(), self.input_bits);
        assert_eq!(output.len(), self.output_bits);

        let mut t = self.root_bit;
        let mut s = self.root;
        let mut ex_node = ExpandedNode::new();
        for (idx, cw) in self.cws.iter().enumerate() {
            let path_bit = x.get(idx);
            ex_node.fill(&s);
            s = if path_bit {
                ex_node.node_r
            } else {
                ex_node.node_l
            };
            if t {
                s ^= &cw.node;
            }
            let (new_t, t_corr) = if path_bit {
                (ex_node.get_right_bit(), cw.bit_r)
            } else {
                (ex_node.get_left_bit(), cw.bit_l)
            };
            t = new_t ^ (t & t_corr);
        }
        if x.len() > self.cws.len() {
            let mut output_node = Node::default();
            let levels_packed = x.len() - self.cws.len();
            let mut bs = BitSliceMut::new(
                self.output_bits << levels_packed,
                std::slice::from_mut(&mut output_node),
            );
            convert_into(&s, &mut bs);
            if t {
                bs.bitxor_assign(&self.last_cw);
            }
            let idx_in_block = {
                let mut pos = 0;
                for i in self.cws.len()..x.len() {
                    pos <<= 1;
                    pos ^= (x.get(i)) as usize;
                }
                pos
            };
            let start = idx_in_block * self.output_bits;
            for i in 0..self.output_bits {
                output.set(i, bs.get(start + i));
            }
        } else {
            convert_into(&s, output);
            if t {
                output.bitxor_assign(&self.last_cw);
            }
        }
    }
    // This is recommended if there's a small number of bits in the output an so it generates a very compact output.
    pub fn eval_all_bits(&self) -> Vec<Node> {
        const BATCH_SIZE: usize = 1;
        let (tree_depth, leaf_depth) = tree_and_leaf_depth(self.input_bits, self.output_bits);
        if self.input_bits == 0 {
            return Vec::new();
        }
        let total_items = 1 << (self.input_bits - leaf_depth);
        let mut output_vec = Vec::with_capacity(total_items);
        let mut output_container = BitVec::new(self.output_bits << leaf_depth);
        let mut output_container_bitslice_mut = BitSliceMut::from(&mut output_container);
        let mut container = vec![Node::default(); BATCH_SIZE * 3];
        let mut t = self.root_bit;
        let mut s = self.root;
        let mut next_directions = Vec::with_capacity(self.input_bits - leaf_depth);
        'outer: loop {
            let depth = next_directions.len();
            if depth == tree_depth {
                convert_into(&s, &mut output_container_bitslice_mut);
                if t {
                    output_container_bitslice_mut.bitxor_assign(&self.last_cw);
                }
                output_container_bitslice_mut
                    .v
                    .iter()
                    .for_each(|v| output_vec.push(*v));
                loop {
                    let (dir, t_r, s_r): (bool, bool, Node) = match next_directions.pop() {
                        None => break 'outer,
                        Some(tuple) => tuple,
                    };
                    if !dir {
                        // We finished left, go to right
                        t = t_r;
                        s = s_r;
                        next_directions.push((true, t, s));
                        break;
                    }
                    // Otherwise we finished right, keep going up
                }
            } else {
                let [mut s_l, mut s_r, bits] = triple_prg(&s);
                let mut t_l = bits.get_bit(0);
                let mut t_r = bits.get_bit(1);
                if t {
                    s_l ^= &self.cws[depth].node;
                    s_r ^= &self.cws[depth].node;
                    t_l ^= self.cws[depth].bit_l;
                    t_r ^= self.cws[depth].bit_r;
                }
                next_directions.push((false, t_r, s_r));
                s = s_l;
                t = t_l;
            }
        }
        output_vec
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
    use std::ops::BitXorAssign;

    use aes_prng::AesRng;
    use rand::RngCore;

    use crate::dpf::{int_to_bits, BitSliceMut, BitVec};

    use super::{DpfKey, Node};

    #[test]
    fn test_dpf() {
        const DEPTH: usize = 10;
        const OUTPUT_WIDTH: usize = 128;
        let mut rng = AesRng::from_random_seed();
        let root_0 = Node::random(&mut rng);
        let root_1 = Node::random(&mut rng);
        let roots = (root_0, root_1);
        let alpha_idx = (rng.next_u32() as usize) & ((1 << DEPTH) - 1);
        let alpha: Vec<_> = int_to_bits(alpha_idx, DEPTH);
        let alpha_v = BitVec::from(&alpha[..]);
        let beta: Vec<bool> = (0..OUTPUT_WIDTH).map(|_| rng.next_u32() & 1 == 1).collect();
        let beta_bitvec = BitVec::from(&beta[..]);
        let (k_0, k_1) = DpfKey::gen(roots, alpha_v, (&beta[..]).into());
        let mut output_0 = BitVec::new(OUTPUT_WIDTH);
        let mut output_1 = BitVec::new(OUTPUT_WIDTH);
        for i in 0usize..1 << DEPTH {
            let bits_i = int_to_bits(i, DEPTH);
            let input = BitVec::from(&bits_i[..]);
            let mut bs_output_0 = BitSliceMut::from(&mut output_0);
            let mut bs_output_1 = BitSliceMut::from(&mut output_1);
            k_0.eval(&input, &mut bs_output_0);
            k_1.eval(&input, &mut bs_output_1);
            if i != alpha_idx {
                assert_eq!(&bs_output_0, &bs_output_1);
            }
            if i == alpha_idx {
                output_0.bitxor_assign(&output_1);
                assert_eq!(output_0, beta_bitvec);
            }
        }
    }

    #[test]
    fn test_dpf_evalall() {
        const DEPTH: usize = 10;
        const OUTPUT_WIDTH: usize = 65;
        let mut rng = AesRng::from_random_seed();
        let root_0 = Node::random(&mut rng);
        let root_1 = Node::random(&mut rng);
        let roots = (root_0, root_1);
        let alpha_idx = (rng.next_u32() as usize) & ((1 << DEPTH) - 1);
        let alpha: Vec<_> = int_to_bits(alpha_idx, DEPTH);
        let alpha_v = BitVec::from(&alpha[..]);
        let beta: Vec<bool> = (0..OUTPUT_WIDTH).map(|_| rng.next_u32() & 1 == 1).collect();
        let beta_bitvec = BitVec::from(&beta[..]);
        let (k_0, k_1) = DpfKey::gen(roots, alpha_v, (&beta[..]).into());
        let ev_all_0 = k_0.eval_all_bits();
        let ev_all_1 = k_1.eval_all_bits();
        for i in 0usize..1 << DEPTH {
            let bits_i = int_to_bits(i, DEPTH);
            let input = BitVec::from(&bits_i[..]);
            if i != alpha_idx {
                assert_eq!(&ev_all_0[i], &ev_all_1[i]);
            }
            if i == alpha_idx {
                let mut xors = ev_all_1[i];
                xors.bitxor_assign(&ev_all_0[i]);
                let t = beta_bitvec.v[0];
                assert_eq!(t, xors);
            }
        }
    }
}
