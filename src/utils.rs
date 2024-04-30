use std::{
    cmp::Ordering,
    iter::Sum,
    ops::{
        Add, AddAssign, BitXor, BitXorAssign, Div, Index, IndexMut, Mul, MulAssign, Neg, Sub,
        SubAssign,
    },
};

use crate::{field::FieldElement, rb_okvs::binary_okvs::BinaryOkvsValue};
use crate::{
    rb_okvs::{OkvsKey, OkvsValue},
    DpfOutput,
};
use rand::{CryptoRng, RngCore};

use crate::{prg::many_prg, BITS_OF_SECURITY, BYTES_OF_SECURITY};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, Hash)]
pub struct Node(u128);
impl From<u128> for Node {
    fn from(value: u128) -> Self {
        Self(value)
    }
}
impl BinaryOkvsValue for Node {
    fn random<R: CryptoRng + RngCore>(rng: R) -> Self {
        Node::random(rng)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}
impl DpfOutput for Node {}
impl Neg for Node {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self
    }
}
impl Add for Node {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self ^ rhs
    }
}
impl Sub for Node {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self ^ rhs
    }
}
impl OkvsKey for Node {
    fn hash_seed(&self, base_seed: &[u8; 16]) -> [u8; 16] {
        let my_bytes = self.0.to_le_bytes();
        core::array::from_fn(|i| my_bytes[i] ^ base_seed[i])
    }
    fn random<R: CryptoRng + RngCore>(rng: R) -> Self {
        Self::random(rng)
    }
}
impl From<Node> for u128 {
    fn from(value: Node) -> Self {
        value.0
    }
}
impl AsRef<u128> for Node {
    fn as_ref(&self) -> &u128 {
        &self.0
    }
}
impl OkvsValue for Node {
    fn random<R: CryptoRng + RngCore>(rng: R) -> Self {
        let mut output = Node::default();
        output.fill(rng);
        output
    }
    fn is_zero(&self) -> bool {
        self.0 == 0u128
    }
    fn inv(&self) -> Self {
        assert_eq!(self.0, 1u128);
        self.clone()
    }
}
impl MulAssign for Node {
    fn mul_assign(&mut self, rhs: Self) {
        // assert_eq!(self.0, 1u128);
        self.0 *= rhs.0;
    }
}
impl Mul for Node {
    type Output = Self;
    fn mul(mut self, rhs: Self) -> Self::Output {
        // assert_eq!(self.0, 1u128);
        self *= rhs;
        self
    }
}
impl AddAssign for Node {
    fn add_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl SubAssign for Node {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0
    }
}
impl From<bool> for Node {
    fn from(value: bool) -> Self {
        Node(value as u128)
    }
}
impl Div for Node {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        assert_eq!(rhs.0, 1u128);
        self
    }
}
impl Sum for Node {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Node::default(), |cur, acc| Node(cur.0 ^ acc.0))
    }
}
impl Mul<bool> for Node {
    type Output = Self;
    fn mul(self, rhs: bool) -> Self::Output {
        Self(self.0 * (rhs as u128))
    }
}
impl BitXorAssign for Node {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Node {
    pub fn zero(&mut self) {
        self.0 = 0;
    }
    pub fn random<R: CryptoRng + RngCore>(rng: R) -> Self {
        let mut output = Node::default();
        output.fill(rng);
        output
    }
    pub fn pop_first_two_bits(&mut self) -> (bool, bool) {
        let output = self.0 & 1 == 1;
        let output_2 = self.0 & 2 == 2;
        self.0 &= !3u128;
        // let v = self.0.overflowing_shr(1).0;
        // let v = v.overflowing_shr(1).0;
        // self.0 = v;
        (output, output_2)
    }
    pub fn push_first_two_bits(&mut self, bit_1: bool, bit_2: bool) {
        let first = bit_1 as u128;
        let second = (bit_2 as u128) << 1;
        self.0 = (self.0) ^ first ^ second;
    }
    pub fn fill(&mut self, mut rng: impl RngCore + CryptoRng) {
        let ptr = self.as_mut();
        rng.fill_bytes(ptr);
    }
    pub fn shl(&mut self, amount: u32) {
        self.0 = self.0.overflowing_shl(amount).0;
    }
    pub fn shr(&mut self, amount: u32) {
        self.0 = self.0.overflowing_shr(amount).0;
    }
    pub fn get_bit(&self, index: usize) -> bool {
        let index = 127 - index;
        self.0 >> index & 1 == 1
    }
    pub fn get_bit_lsb(&self, index: usize) -> bool {
        ((self.0 >> index) & 1) == 1
    }

    pub fn set_bit(&mut self, index: usize, value: bool) {
        // let (cell, bit) = Self::coordinates(index);
        let index = 127 - index;
        let mask: u128 = !(1 << index);
        self.0 = (self.0 & mask) ^ ((value as u128) << index)
    }
    pub fn set_bit_lsb(&mut self, index: usize, value: bool) {
        // let (cell, bit) = Self::coordinates(index);
        let mask: u128 = !(1 << index);
        self.0 = (self.0 & mask) ^ ((value as u128) << index)
    }
    pub fn toggle_bit(&mut self, index: usize) {
        // let (cell, bit) = Self::coordinates(index);
        let index = 127 - index;
        self.0 ^= 1 << index;
    }
    pub fn mask(&mut self, bits: usize) {
        let mask = ((1u128 << bits) - 1) << ((BITS_OF_SECURITY - bits) & (BITS_OF_SECURITY - 1));
        self.0 &= mask;
    }
    pub fn mask_lsbs(&mut self, bits: usize) {
        let mask = !0u128 >> (128 - bits);
        // let mask = (1 << bits) - 1;
        self.0 &= mask;
    }
    pub fn mask_bits_lsbs(&mut self, from: usize, to: usize) {
        let mask = ((!0u128) >> (128 - to + from)) << from;
        self.0 &= !mask;
    }
    pub fn cmp_first_bits(&self, other: &Self, i: usize) -> Ordering {
        if i >= BITS_OF_SECURITY {
            return self.cmp(other);
        }
        let mask = (1 << i) - 1;
        let self_first = self.0 & mask;
        let other_first = other.0 & mask;
        self_first.cmp(&other_first)
    }
}
impl BitXor for Node {
    type Output = Node;
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl Default for Node {
    fn default() -> Self {
        Self(0u128)
    }
}

impl BitXor<&Node> for &Node {
    type Output = Node;
    fn bitxor(self, rhs: &Node) -> Self::Output {
        Node(self.0 ^ rhs.0)
    }
}
impl BitXorAssign<&Node> for Node {
    fn bitxor_assign(&mut self, rhs: &Node) {
        self.0 ^= rhs.0;
    }
}
impl AsRef<[u8; 16]> for Node {
    fn as_ref(&self) -> &[u8; 16] {
        unsafe {
            (&self.0 as *const u128 as *const [u8; 16])
                .as_ref()
                .unwrap()
        }
    }
}
impl AsMut<[u8]> for Node {
    fn as_mut(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(&mut self.0 as *mut u128 as *mut u8, BYTES_OF_SECURITY)
        }
    }
}

pub trait SmallFieldContainer<const SIZE: usize, F: FieldElement>:
    DpfOutput
    + From<[F; SIZE]>
    + Into<[F; SIZE]>
    + Clone
    + Copy
    + Index<usize, Output = F>
    + IndexMut<usize, Output = F>
    + OkvsValue
    + Default
{
    fn zero() -> Self {
        Self::from([F::zero(); SIZE])
    }
}

#[derive(Clone)]
pub struct ExpandedNode {
    nodes: Box<[Node]>,
}
impl ExpandedNode {
    pub fn new(non_zero_point_count: usize) -> Self {
        let node_count = 2 + 2 * ((non_zero_point_count + BITS_OF_SECURITY - 1) / BITS_OF_SECURITY);
        ExpandedNode {
            nodes: Box::from(vec![Node::default(); node_count]),
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
#[derive(Clone, PartialEq, Eq, Debug, Hash, PartialOrd)]
pub struct BitVec {
    pub(crate) v: Box<[Node]>,
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
// Lexicographic ordering
impl Ord for BitVec {
    fn cmp(&self, other: &Self) -> Ordering {
        for (me_cell, other_cell) in self.v.iter().zip(other.v.iter()) {
            match me_cell.cmp(other_cell) {
                Ordering::Equal => continue,
                n @ _ => return n,
            }
        }
        return self.len.cmp(&other.len);
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
    pub fn fill_random(&mut self, mut rng: impl RngCore + CryptoRng) {
        self.v.iter_mut().for_each(|v| v.fill(&mut rng));
        self.normalize();
    }
    pub fn zero(&mut self) {
        self.v.iter_mut().for_each(|v| {
            v.0 = 0;
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
        Self::new(value.len(), &value.as_ref())
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
pub struct BitSliceMut<'a> {
    v: &'a mut [Node],
    len: usize,
}
impl<'a> BitSliceMut<'a> {
    pub fn new(len: usize, v: &'a mut [Node]) -> Self {
        BitSliceMut { v: v.as_mut(), len }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn get(&self, index: usize) -> bool {
        let (cell, idx) = BitVec::coordinates(index);
        self.v[cell].get_bit(idx)
    }
    pub fn set(&mut self, index: usize, val: bool) {
        let (cell, idx) = BitVec::coordinates(index);
        self.v[cell].set_bit(idx, val);
    }
}
impl<'a> AsMut<[Node]> for BitSliceMut<'a> {
    fn as_mut(&mut self) -> &mut [Node] {
        self.v
    }
}
impl<'a> BitXorAssign<&BitSliceMut<'a>> for BitSliceMut<'a> {
    fn bitxor_assign(&mut self, rhs: &BitSliceMut<'a>) {
        assert_eq!(self.len, rhs.len);
        for i in 0..self.v.len() {
            self.v[i].bitxor_assign(&rhs.v[i]);
        }
    }
}
impl<'a> BitXorAssign<&BitVec> for BitSliceMut<'a> {
    fn bitxor_assign(&mut self, rhs: &BitVec) {
        assert_eq!(self.len, rhs.len);
        for i in 0..self.v.len() {
            self.v[i].bitxor_assign(&rhs.v[i]);
        }
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
#[derive(Debug, PartialEq, Eq)]
pub struct BitSlice<'a> {
    v: &'a [Node],
    len: usize,
}
impl<'a> BitSlice<'a> {
    pub fn new(len: usize, v: &'a [Node]) -> Self {
        BitSlice { v: v.as_ref(), len }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn get(&self, index: usize) -> bool {
        let (cell, idx) = BitVec::coordinates(index);
        self.v[cell].get_bit(idx)
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
        assert_eq!(self.len, rhs.len);
        for i in 0..self.v.len() {
            self.v[i].bitxor_assign(&rhs.v[i]);
        }
    }
}
