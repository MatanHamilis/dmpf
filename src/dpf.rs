use once_cell::sync::Lazy;
use std::cmp::Ordering;
use std::ops::BitXorAssign;

use crate::xor_arrays;

use super::BITS_OF_SECURITY;
use super::BYTES_OF_SECURITY;
use aes::{
    cipher::{BlockEncrypt, KeyInit},
    Aes128, Block,
};
use rand::{CryptoRng, RngCore};
const DPF_AES_KEY: [u8; BYTES_OF_SECURITY] =
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2];
static AES: Lazy<Aes128> = Lazy::new(|| Aes128::new_from_slice(&DPF_AES_KEY).unwrap());

pub fn double_prg(input: &Node, children: &[u8; 2]) -> [Node; 2] {
    let mut blocks = [Block::from(*input.as_ref()); 2];
    blocks[0][0] ^= children[0];
    blocks[1][0] ^= children[1];
    AES.encrypt_blocks(&mut blocks);
    xor_arrays(&mut blocks[0].into(), input.as_ref());
    xor_arrays(&mut blocks[1].into(), input.as_ref());
    blocks[0][0] ^= children[0];
    blocks[1][0] ^= children[1];
    unsafe { std::mem::transmute(blocks) }
}
pub fn triple_prg(input: &Node, children: &[u8; 3]) -> [Node; 3] {
    let mut blocks = [Block::from(*input.as_ref()); 3];
    blocks[0][0] ^= children[0];
    blocks[1][0] ^= children[1];
    blocks[2][0] ^= children[2];
    AES.encrypt_blocks(&mut blocks);
    xor_arrays(&mut blocks[0].into(), input.as_ref());
    xor_arrays(&mut blocks[1].into(), input.as_ref());
    xor_arrays(&mut blocks[2].into(), input.as_ref());
    blocks[0][0] ^= children[0];
    blocks[1][0] ^= children[1];
    blocks[2][0] ^= children[2];
    unsafe { std::mem::transmute(blocks) }
}
pub fn many_prg(
    input: &Node,
    mut children: impl Iterator<Item = u16> + Clone,
    output: &mut [Node],
) {
    let input_block = Block::from(*input.as_ref());
    if output.len() > 256 {
        unimplemented!()
    }
    let mut blocks_output =
        unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut Block, output.len()) };
    let children_copy = children.clone();
    blocks_output
        .iter_mut()
        .zip(children_copy)
        .for_each(|(mut v, child)| {
            *v = input_block;
            let bytes = child.to_be_bytes();
            v[0] ^= bytes[0];
            v[1] ^= bytes[1];
        });
    AES.encrypt_blocks(&mut blocks_output);
    blocks_output
        .iter_mut()
        .zip(children)
        .for_each(|(mut v, child)| {
            let bytes = child.to_be_bytes();
            v[0] ^= bytes[0];
            v[1] ^= bytes[1];
        });
}

pub struct NodeBitStructs<'a> {
    node: &'a mut Node,
    bit_width: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, Hash)]
pub struct Node([u64; BYTES_OF_SECURITY / 8]);
impl Node {
    pub fn zero(&mut self) {
        self.0[0] = 0;
        self.0[1] = 0;
    }
    pub fn random<R: CryptoRng + RngCore>(rng: R) -> Self {
        let mut output = Node::default();
        output.fill(rng);
        output
    }
    pub fn fill(&mut self, mut rng: impl RngCore + CryptoRng) {
        let ptr = self.as_mut();
        rng.fill_bytes(ptr);
    }
    fn coordinates(index: usize) -> (usize, usize) {
        (
            index / u64::BITS as usize,
            63 - (index & (u64::BITS as usize - 1)),
        )
    }
    pub fn shl(&mut self, mut amount: u32) {
        if amount >= 64 {
            self.0[0] = self.0[1];
            self.0[1] = 0;
            amount -= 64;
        }
        if amount == 0 {
            return;
        }
        self.0[0] = self.0[0].overflowing_shl(amount).0;
        self.0[0] |= self.0[1].overflowing_shr(64 - amount).0;
        self.0[1] = self.0[1].overflowing_shl(amount).0;
    }
    pub fn shr(&mut self, mut amount: u32) {
        if amount >= 64 {
            self.0[1] = self.0[0];
            self.0[0] = 0;
            amount -= 64;
        }
        if amount == 0 {
            return;
        }
        self.0[1] = self.0[1].overflowing_shr(amount).0;
        self.0[1] |= self.0[0].overflowing_shl(64 - amount).0;
        self.0[0] = self.0[0].overflowing_shr(amount).0;
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
    pub fn toggle_bit(&mut self, index: usize) {
        let (cell, bit) = Self::coordinates(index);
        self.0[cell] ^= 1 << bit;
    }
    pub fn mask(&mut self, bits: usize) {
        self.shr((BITS_OF_SECURITY - bits) as u32);
        self.shl((BITS_OF_SECURITY - bits) as u32);
    }
    pub fn cmp_first_bits(&self, other: &Self, i: usize) -> Ordering {
        if i >= BITS_OF_SECURITY {
            return self.cmp(other);
        }
        let first_cell_bits = std::cmp::min(64, i);
        let self_first = self.0[0] >> (64 - first_cell_bits);
        let other_first = other.0[0] >> (64 - first_cell_bits);
        match self_first.cmp(&other_first) {
            Ordering::Equal => {}
            result @ _ => return result,
        }
        let second_cell_bits = i - first_cell_bits;
        let self_first = self.0[0] >> (64 - second_cell_bits);
        let other_first = other.0[0] >> (64 - second_cell_bits);
        self_first.cmp(&other_first)
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        for i in 0..self.0.len() {
            match self.0[i].cmp(&other.0[i]) {
                Ordering::Equal => continue,
                result @ _ => return Some(result),
            }
        }
        Some(Ordering::Equal)
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
pub(crate) struct ExpandedNode {
    pub nodes: Box<[Node]>,
    pub non_zero_point_count: usize,
}
impl ExpandedNode {
    pub fn new(non_zero_point_count: usize) -> Self {
        let node_count = 2 + 2 * ((non_zero_point_count + BITS_OF_SECURITY - 1) / BITS_OF_SECURITY);
        ExpandedNode {
            nodes: Box::from(vec![Node::default(); node_count]),
            non_zero_point_count,
        }
    }
    pub fn fill_all(&mut self, node: &Node) {
        many_prg(node, 0..self.nodes.len() as u16, &mut self.nodes);
    }
    pub fn fill(&mut self, node: &Node, direction: bool) {
        let start = direction as u16;
        let end = start + self.nodes.len() as u16 - 1;
        many_prg(
            node,
            start..end,
            &mut self.nodes[start as usize..end as usize],
        );
    }
    pub fn get_bit(&self, idx: usize, direction: bool) -> bool {
        let bits = self.get_bits_direction(direction);
        let node = idx / BITS_OF_SECURITY;
        let bit_idx = idx & (BITS_OF_SECURITY - 1);
        bits[node].get_bit(bit_idx)
    }
    pub fn get_node(&self, direction: bool) -> &Node {
        &self.nodes[(direction as usize) * (self.nodes.len() - 1)]
    }
    pub fn get_node_mut(&mut self, direction: bool) -> &mut Node {
        &mut self.nodes[(direction as usize) * (self.nodes.len() - 1)]
    }
    pub fn get_bits(&self) -> &[Node] {
        &self.nodes[1..self.nodes.len() - 1]
    }
    pub fn get_bits_mut(&mut self) -> &mut [Node] {
        let len = self.nodes.len();
        &mut self.nodes[1..len - 1]
    }
    pub fn get_bits_direction(&self, direction: bool) -> &[Node] {
        let direction_len = (self.nodes.len() - 2) / 2;
        let node = 1 + (direction as usize) * direction_len;
        &self.nodes[node..node + direction_len]
    }
}
#[derive(Clone, PartialEq, Eq, Debug, Hash, Ord)]
pub struct BitVec {
    v: Box<[Node]>,
    len: usize,
}
impl AsRef<[Node]> for BitVec {
    fn as_ref(&self) -> &[Node] {
        &self.v
    }
}
impl AsMut<[Node]> for BitVec {
    fn as_mut(&mut self) -> &mut [Node] {
        &mut self.v
    }
}
impl PartialOrd for BitVec {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.len() != other.len() {
            return None;
        }
        for (self_node, node) in self.v.iter().zip(other.v.iter()) {
            match self_node.cmp(node) {
                Ordering::Equal => continue,
                result @ _ => return Some(result),
            }
        }
        Some(Ordering::Equal)
    }
}

impl BitVec {
    pub fn fill(&mut self, nodes: &[Node]) {
        self.v.copy_from_slice(nodes);
        // self.normalize();
    }
    pub fn new(len: usize) -> Self {
        let nodes = (len + BITS_OF_SECURITY - 1) / BITS_OF_SECURITY;
        let v = vec![Node::default(); nodes].into();
        BitVec { v, len }
    }
    pub fn zero(&mut self) {
        self.v.iter_mut().for_each(|v| {
            v.0[0] = 0;
            v.0[1] = 0;
        });
    }
    pub fn len(&self) -> usize {
        self.len
    }
    fn coordinates(index: usize) -> (usize, usize) {
        (index / BITS_OF_SECURITY, index & (BITS_OF_SECURITY - 1))
    }
    pub fn get(&self, index: usize) -> bool {
        let (cell, idx) = Self::coordinates(index);
        self.v[cell].get_bit(idx)
    }
    pub fn set(&mut self, index: usize, val: bool) {
        let (cell, idx) = Self::coordinates(index);
        self.v[cell].set_bit(idx, val);
    }
    pub(crate) fn normalize(&mut self) {
        let bits_to_leave = self.len & (BITS_OF_SECURITY - 1);
        let bits_to_leave = ((bits_to_leave + BITS_OF_SECURITY - 1) % BITS_OF_SECURITY) + 1;
        self.v.last_mut().unwrap().mask(bits_to_leave);
    }
}
impl<'a> From<&'a BitVec> for BitSlice<'a> {
    fn from(value: &'a BitVec) -> Self {
        Self::new(value.len(), value.as_ref())
    }
}
impl<'a> PartialOrd for BitSlice<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.len() != other.len() {
            return None;
        }
        for (self_node, node) in self.v.iter().zip(other.v.iter()) {
            match self_node.cmp(node) {
                Ordering::Equal => continue,
                result @ _ => return Some(result),
            }
        }
        Some(Ordering::Equal)
    }
}
#[derive(Debug, PartialEq, Eq)]
pub struct BitSlice<'a> {
    v: &'a [Node],
    len: usize,
}
impl<'a> BitSlice<'a> {
    pub fn new(len: usize, v: &'a [Node]) -> Self {
        BitSlice { v, len }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    fn coordinates(index: usize) -> (usize, usize) {
        (index / BITS_OF_SECURITY, index & (BITS_OF_SECURITY - 1))
    }
    pub fn get(&self, index: usize) -> bool {
        let (cell, idx) = Self::coordinates(index);
        self.v[cell].get_bit(idx)
    }
    fn shrink(&mut self, new_len: usize) {
        assert!(new_len <= self.len);
        self.len = new_len;
    }
}

// #[derive(Debug, PartialEq, Eq)]
// pub struct BitSliceMut<'a> {
//     v: &'a mut [Node],
//     len: usize,
// }
// impl<'a> AsMut<[Node]> for BitSliceMut<'a> {
//     fn as_mut(&mut self) -> &mut [Node] {
//         &mut self.v
//     }
// }
// impl<'a> BitSliceMut<'a> {
//     pub fn new(len: usize, v: &'a mut [Node]) -> Self {
//         BitSliceMut { v, len }
//     }
//     pub fn len(&self) -> usize {
//         self.len
//     }
//     fn coordinates(index: usize) -> (usize, usize) {
//         (index / BITS_OF_SECURITY, index & (BITS_OF_SECURITY - 1))
//     }
//     pub fn get(&self, index: usize) -> bool {
//         let (cell, idx) = Self::coordinates(index);
//         self.v[cell].get_bit(idx)
//     }
//     pub fn set(&mut self, index: usize, val: bool) {
//         let (cell, idx) = Self::coordinates(index);
//         self.v[cell].set_bit(idx, val);
//     }
//     pub fn fill(&mut self, nodes: &[Node]) {
//         self.v.copy_from_slice(nodes);
//         self.normalize();
//     }
//     pub(crate) fn normalize(&mut self) {
//         let bits_to_leave = self.len & (BITS_OF_SECURITY - 1);
//         let bits_to_leave = ((bits_to_leave + BITS_OF_SECURITY - 1) % BITS_OF_SECURITY) + 1;
//         self.v.last_mut().unwrap().mask(bits_to_leave);
//     }
// }
// impl<'a> From<&'a mut BitVec> for BitSliceMut<'a> {
//     fn from(value: &'a mut BitVec) -> Self {
//         Self {
//             v: &mut value.v,
//             len: value.len,
//         }
//     }
// }

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
        assert_eq!(self.len, rhs.len);
        for i in 0..self.v.len() {
            self.v[i].bitxor_assign(&rhs.v[i]);
        }
    }
}
// impl<'a> BitXorAssign<&BitVec> for BitSliceMut<'a> {
//     fn bitxor_assign(&mut self, rhs: &BitVec) {
//         assert_eq!(self.len, rhs.len);
//         let bytes = (self.len + BITS_OF_SECURITY - 1) / BITS_OF_SECURITY;
//         for i in 0..bytes {
//             self.v[i].bitxor_assign(&rhs.v[i]);
//         }
//     }
// }
#[derive(Clone, Copy)]
pub struct CorrectionWord {
    node: Node,
    bits: u8,
}
pub struct DpfKey {
    root: Node,
    root_bit: bool,
    cws: Vec<CorrectionWord>,
    last_cw: BitVec,
    input_bits: usize,
    output_bits: usize,
}
impl CorrectionWord {
    fn get_bit(&self, direction: bool) -> bool {
        self.bits & (1 << (direction as u8)) != 0
    }
    fn new(node: Node, left_bit: bool, right_bit: bool) -> Self {
        Self {
            node,
            bits: (left_bit as u8) + 2 * (right_bit as u8),
        }
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
    convert_into(node, &mut output);
    output
}
pub(crate) fn convert_into(node: &Node, output: &mut BitVec) {
    let bits = output.len;
    if bits > BITS_OF_SECURITY {
        // We should expand node
        let nodes_num: u16 = ((output.len + BITS_OF_SECURITY - 1) / BITS_OF_SECURITY) as u16;
        many_prg(node, 0..nodes_num as u16, output.as_mut());
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
        let mut expanded_node_0 = ExpandedNode::new(1);
        let mut expanded_node_1 = ExpandedNode::new(1);
        let mut seed_0 = roots.0;
        let mut seed_1 = roots.1;
        let (tree_depth, leaf_depth) = tree_and_leaf_depth(alpha.len(), beta.len());
        let mut cws = Vec::with_capacity(tree_depth);
        for i in 0..tree_depth {
            expanded_node_0.fill_all(&seed_0);
            expanded_node_1.fill_all(&seed_1);
            let path_bit = alpha.get(i);
            let t_l_0 = expanded_node_0.get_bit(0, false);
            let t_l_1 = expanded_node_1.get_bit(0, false);
            let t_r_0 = expanded_node_0.get_bit(0, true);
            let t_r_1 = expanded_node_1.get_bit(0, true);
            let bit_left = !(t_l_0 ^ t_l_1 ^ path_bit);
            let bit_right = t_r_0 ^ t_r_1 ^ path_bit;
            expanded_node_0
                .get_node_mut(!path_bit)
                .bitxor_assign(expanded_node_1.get_node(!path_bit));
            seed_0 = *expanded_node_0.get_node(path_bit);
            seed_1 = *expanded_node_1.get_node(path_bit);
            let cw_node = *expanded_node_0.get_node(!path_bit);
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
            cws.push(CorrectionWord::new(cw_node, bit_left, bit_right));
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
        convert_into(&seed_1, &mut conv);
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
    pub fn eval(&self, x: &BitVec, output: &mut BitVec) {
        assert_eq!(x.len(), self.input_bits);
        assert_eq!(output.len(), self.output_bits);

        let mut t = self.root_bit;
        let mut s = self.root;
        let mut ex_node = ExpandedNode::new(1);
        for (idx, cw) in self.cws.iter().enumerate() {
            let path_bit = x.get(idx);
            ex_node.fill_all(&s);
            s = *ex_node.get_node(path_bit);
            if t {
                s ^= &cw.node;
            }
            let (new_t, t_corr) = (ex_node.get_bit(0, path_bit), cw.get_bit(path_bit));
            t = new_t ^ (t & t_corr);
        }
        // let mut output_node = Node::default();
        // let mut bs = BitSliceMut::new(
        //     self.output_bits << levels_packed,
        //     std::slice::from_mut(&mut output_node),
        // );
        if x.len() > self.cws.len() {
            let levels_packed = x.len() - self.cws.len();
            let mut bs = BitVec::new(self.output_bits << levels_packed);
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
    pub fn eval_all(&self) -> EvalAllResult {
        const BATCH_SIZE: usize = 1;
        let (tree_depth, leaf_depth) = tree_and_leaf_depth(self.input_bits, self.output_bits);
        let blocks_per_output = (self.output_bits + BITS_OF_SECURITY - 1) / BITS_OF_SECURITY;
        let total_items = 1 << (self.input_bits - leaf_depth);
        let mut output_vec = Vec::with_capacity(total_items);
        let mut output_container = BitVec::new(self.output_bits << leaf_depth);
        let mut container = vec![Node::default(); BATCH_SIZE * 3];
        let mut t = self.root_bit;
        let mut s = self.root;
        let mut expanded_node = ExpandedNode::new(1);

        let mut next_directions = Vec::with_capacity(self.input_bits - leaf_depth);
        'outer: loop {
            let depth = next_directions.len();
            if depth == tree_depth {
                convert_into(&s, &mut output_container);
                if t {
                    output_container.bitxor_assign(&self.last_cw);
                }
                output_container.v.iter().for_each(|v| output_vec.push(*v));
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
                expanded_node.fill_all(&s);
                let mut s_l = *expanded_node.get_node(false);
                let mut s_r = *expanded_node.get_node(true);
                let mut t_l = expanded_node.get_bit(0, false);
                let mut t_r = expanded_node.get_bit(0, true);
                if t {
                    s_l ^= &self.cws[depth].node;
                    s_r ^= &self.cws[depth].node;
                    t_l ^= self.cws[depth].get_bit(false);
                    t_r ^= self.cws[depth].get_bit(true);
                }
                next_directions.push((false, t_r, s_r));
                s = s_l;
                t = t_l;
            }
        }
        EvalAllResult {
            nodes: output_vec,
            output_bits: self.output_bits,
            outputs_per_block: 1 << leaf_depth,
            blocks_per_output,
        }
    }
}
pub struct EvalAllResult {
    nodes: Vec<Node>,
    output_bits: usize,
    blocks_per_output: usize,
    outputs_per_block: usize,
}
impl EvalAllResult {
    pub fn get_item(&self, i: usize, output: &mut [Node]) {
        if self.output_bits == 0 {
            return;
        }
        assert_eq!(output.len(), self.blocks_per_output);
        let coord = i * self.blocks_per_output / self.outputs_per_block;
        let idx_in_block = i % self.outputs_per_block;
        let start_bit = idx_in_block * self.output_bits;
        output.copy_from_slice(&self.nodes[coord..coord + self.blocks_per_output]);
        if self.outputs_per_block > 1 {
            // clearing other bits
            output[0].shr((BITS_OF_SECURITY - start_bit - self.output_bits) as u32);
            output[0].shl((BITS_OF_SECURITY - self.output_bits) as u32);
        }
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
    use rand::{thread_rng, RngCore};

    use crate::dpf::{int_to_bits, tree_and_leaf_depth, BitVec};

    use super::{DpfKey, Node};

    #[test]
    fn test_dpf() {
        const DEPTH: usize = 10;
        const OUTPUT_WIDTH: usize = 2;
        let mut rng = thread_rng();
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
            let mut bs_output_0 = &mut output_0;
            let mut bs_output_1 = &mut output_1;
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
        const DEPTH: usize = 8;
        const OUTPUT_WIDTH: usize = 3;
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
        let ev_all_0 = k_0.eval_all();
        let ev_all_1 = k_1.eval_all();
        let mut output_0 = vec![Node::default(); ev_all_0.blocks_per_output];
        let mut output_1 = vec![Node::default(); ev_all_1.blocks_per_output];
        for i in 0usize..1 << DEPTH {
            ev_all_0.get_item(i, &mut output_0);
            ev_all_1.get_item(i, &mut output_1);
            if i != alpha_idx {
                assert_eq!(output_0, output_1);
            } else {
                output_0
                    .iter_mut()
                    .zip(output_1.iter())
                    .for_each(|(output, input)| output.bitxor_assign(input));
                assert_eq!(output_0, beta_bitvec.as_ref());
            }
        }
    }
    #[test]
    fn test_bitvec_comparison() {
        const DEPTH: usize = 3;
        for i in 1..1 << DEPTH {
            let cur = int_to_bits(i, DEPTH);
            let prev = int_to_bits(i - 1, DEPTH);
            let cur_bv = BitVec::from(&cur[..]);
            let prev_bv = BitVec::from(&prev[..]);
            assert!(cur_bv > prev_bv);
        }
    }
}
