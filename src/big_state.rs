use std::{
    ops::Div,
    time::{Duration, Instant},
};

use rand::{CryptoRng, RngCore};

use crate::{
    prg::{double_prg, double_prg_many, many_many_prg, many_prg, DOUBLE_PRG_CHILDREN},
    trie::BinaryTrie,
    BitSlice, Dmpf, DmpfKey, DmpfSession, DpfOutput, Node,
};

pub struct BigStateDmpf;
impl BigStateDmpf {
    pub fn new() -> Self {
        Self
    }
}

impl<Output: DpfOutput> Dmpf<Output> for BigStateDmpf {
    type Key = BigStateDmpfKey<Output>;
    fn try_gen<R: rand::prelude::CryptoRng + rand::prelude::RngCore>(
        &self,
        input_length: usize,
        inputs: &[(u128, Output)],
        mut rng: &mut R,
    ) -> Option<(Self::Key, Self::Key)> {
        // Make sure inputs are sorted.
        for i in 0..inputs.len() - 1 {
            assert!(inputs[i].0 < inputs[i + 1].0);
        }

        let mut trie = BinaryTrie::default();
        let t = inputs.len();
        let mut signs_0 = KeyGenSigns::new(t);
        let mut signs_1 = KeyGenSigns::new(t);
        let mut next_signs_0 = KeyGenSigns::new(t);
        let mut next_signs_1 = KeyGenSigns::new(t);
        signs_1.set_bit(0, 0, true);
        let mut seed_0 = vec![Node::default(); t];
        let mut seed_1 = vec![Node::default(); t];
        let mut next_seed_0 = vec![Node::default(); t];
        let mut next_seed_1 = vec![Node::default(); t];
        let root_0 = Node::random(&mut rng);
        let root_1 = Node::random(&mut rng);
        seed_0[0] = root_0;
        seed_1[0] = root_1;
        let mut new_signs_0_left = Signs::new(t);
        let mut new_signs_0_right = Signs::new(t);
        let mut new_signs_1_left = Signs::new(t);
        let mut new_signs_1_right = Signs::new(t);
        let mut new_signs_delta_left = Signs::new(t);
        let mut new_signs_delta_right = Signs::new(t);
        let mut cws = Vec::with_capacity(input_length);

        for input in inputs {
            let n = Node::from(input.0);
            let bitslice = BitSlice::new(input_length, std::slice::from_ref(&n));
            trie.insert(&bitslice);
        }

        for depth in 0..input_length {
            let mut sign_cw = SignsCW::new(t, depth, &mut rng);
            let mut seeds = Vec::with_capacity(t);
            let iter = trie.iter_at_depth(depth);
            // Generate CW
            iter.enumerate().for_each(|(idx, node)| {
                let borrowed = node.borrow();
                let has_left = borrowed.get_son(false).is_some();
                let has_right = borrowed.get_son(true).is_some();
                assert!(has_left | has_right);
                let current_seed_0 = seed_0[idx];
                let current_seed_1 = seed_1[idx];
                let [seed_left_0, seed_right_0] = double_prg(&current_seed_0, &DOUBLE_PRG_CHILDREN);
                let [seed_left_1, seed_right_1] = double_prg(&current_seed_1, &DOUBLE_PRG_CHILDREN);
                let delta_seed_left = seed_left_0 ^ seed_left_1;
                let delta_seed_right = seed_right_0 ^ seed_right_1;

                new_signs_0_left.fill_with_seed(&current_seed_0, 0);
                new_signs_0_right.fill_with_seed(&current_seed_0, 1);
                new_signs_1_left.fill_with_seed(&current_seed_1, 0);
                new_signs_1_right.fill_with_seed(&current_seed_1, 1);
                new_signs_delta_left.xor_into(&new_signs_0_left, &new_signs_1_left);
                new_signs_delta_right.xor_into(&new_signs_0_right, &new_signs_1_right);

                seeds.push(if has_left && has_right {
                    Node::random(&mut rng)
                } else if has_left {
                    delta_seed_right
                } else {
                    assert!(has_right);
                    delta_seed_left
                });
                sign_cw.put_sign(&new_signs_delta_left, false, idx);
                sign_cw.put_sign(&new_signs_delta_right, true, idx);
            });
            let cw = CW::gen_cw(depth, &trie, sign_cw, seeds, &mut rng);

            // Update state
            let mut next_position = 0usize;
            trie.iter_at_depth(depth)
                .enumerate()
                .for_each(|(idx, node)| {
                    let borrowed = node.borrow();
                    let has_left = borrowed.get_son(false).is_some();
                    let has_right = borrowed.get_son(true).is_some();
                    let current_seed_0 = seed_0[idx];
                    let current_seed_1 = seed_1[idx];

                    let [seed_left_0, seed_right_0] =
                        double_prg(&current_seed_0, &DOUBLE_PRG_CHILDREN);
                    let [seed_left_1, seed_right_1] =
                        double_prg(&current_seed_1, &DOUBLE_PRG_CHILDREN);

                    if has_left {
                        new_signs_0_left.fill_with_seed(&current_seed_0, 0);
                        new_signs_1_left.fill_with_seed(&current_seed_1, 0);
                    }
                    if has_right {
                        new_signs_0_right.fill_with_seed(&current_seed_0, 1);
                        new_signs_1_right.fill_with_seed(&current_seed_1, 1);
                    }

                    let correct_seed_0 = cw.correct(
                        signs_0.iter(idx),
                        has_left,
                        has_right,
                        &mut new_signs_0_left,
                        &mut new_signs_0_right,
                    );
                    let correct_seed_1 = cw.correct(
                        signs_1.iter(idx),
                        has_left,
                        has_right,
                        &mut new_signs_1_left,
                        &mut new_signs_1_right,
                    );

                    assert!(has_left | has_right);

                    if has_left {
                        next_seed_0[next_position] = seed_left_0 ^ correct_seed_0;
                        next_seed_1[next_position] = seed_left_1 ^ correct_seed_1;
                        next_signs_0.set_signs(next_position, &new_signs_0_left);
                        next_signs_1.set_signs(next_position, &new_signs_1_left);
                        next_position += 1;
                    }
                    if has_right {
                        next_seed_0[next_position] = seed_right_0 ^ correct_seed_0;
                        next_seed_1[next_position] = seed_right_1 ^ correct_seed_1;
                        next_signs_0.set_signs(next_position, &new_signs_0_right);
                        next_signs_1.set_signs(next_position, &new_signs_1_right);
                        next_position += 1;
                    }
                });
            cws.push(cw);
            (
                signs_0,
                signs_1,
                next_signs_0,
                next_signs_1,
                seed_0,
                seed_1,
                next_seed_0,
                next_seed_1,
            ) = (
                next_signs_0,
                next_signs_1,
                signs_0,
                signs_1,
                next_seed_0,
                next_seed_1,
                seed_0,
                seed_1,
            );
        }
        // Handling output part
        let conv_cw = ConvCW::gen(inputs, &seed_0, &seed_1, &signs_0, &signs_1);
        let key_0 = BigStateDmpfKey {
            root: root_0,
            sign: false,
            cws: cws.clone(),
            conv_cw: conv_cw.clone(),
        };
        let key_1 = BigStateDmpfKey {
            root: root_1,
            sign: true,
            cws,
            conv_cw,
        };
        Some((key_0, key_1))
    }
}

#[derive(Debug, Clone)]
struct ConvCW<Output: DpfOutput>(Vec<Output>);
impl<Output: DpfOutput> ConvCW<Output> {
    // kvs are sorted, so it's ok.
    fn gen(
        kvs: &[(u128, Output)],
        seed_0: &[Node],
        seed_1: &[Node],
        sign_0: &KeyGenSigns,
        _: &KeyGenSigns,
    ) -> Self {
        Self(
            kvs.iter()
                .zip(seed_0.iter())
                .zip(seed_1.iter())
                .enumerate()
                .map(|(idx, ((kv, s0), s1))| {
                    let o1 = Output::from(*s1);
                    let o0 = Output::from(*s0);
                    let cw = o0 - o1 - kv.1;
                    if sign_0.get_bit(idx, idx) {
                        -cw
                    } else {
                        cw
                    }
                })
                .collect(),
        )
    }
    fn conv_correct(&self, sign: &Signs) -> Output {
        let mut o = Output::default();
        for (b, oi) in sign.iter().zip(self.0.iter()) {
            if b {
                o += *oi;
            }
        }
        o
    }
}

// Each CW corrects (Lambda + t) bits.
#[derive(Debug, Clone)]
struct CW {
    seeds: Vec<Node>,
    signs: SignsCW,
}
impl CW {
    fn gen_cw(
        depth: usize,
        points: &BinaryTrie,
        mut signs: SignsCW,
        mut seeds: Vec<Node>,
        mut rng: impl RngCore + CryptoRng,
    ) -> Self {
        let t = signs.0;
        let min = t.min(1 << depth);
        let mut total_points_next = 0;
        let iter = points.iter_at_depth(depth);
        iter.enumerate().for_each(|(idx, node)| {
            let borrowed = node.borrow();
            let has_left = borrowed.get_son(false).is_some();
            let has_right = borrowed.get_son(true).is_some();
            if has_left {
                signs.flip_bit(false, idx, total_points_next);
                total_points_next += 1;
            }
            if has_right {
                signs.flip_bit(true, idx, total_points_next);
                total_points_next += 1;
            }
        });

        // No need to randomize signs, they are already randomized at creation to reach correct size.
        while seeds.len() < min {
            seeds.push(Node::random(&mut rng));
        }

        Self { seeds, signs }
    }
    fn correct(
        &self,
        signs: impl Iterator<Item = bool>,
        has_left: bool,
        has_right: bool,
        output_left: &mut Signs,
        output_right: &mut Signs,
    ) -> Node {
        let t = output_left.1;
        self.correct_bits(
            signs,
            has_left,
            has_right,
            &mut output_left.0,
            &mut output_right.0,
            t,
        )
    }
    fn correct_bits(
        &self,
        signs: impl Iterator<Item = bool>,
        has_left: bool,
        has_right: bool,
        output_left: &mut [Node],
        output_right: &mut [Node],
        t: usize,
    ) -> Node {
        let mut correct_node = Node::default();
        signs
            .enumerate()
            .take(self.seeds.len())
            .for_each(|(idx, bit)| {
                if !bit {
                    return;
                }
                correct_node ^= self.seeds[idx];
                if has_left {
                    let coordinates = self.signs.coordinates(false, idx);
                    xor_bits(&self.signs.1[coordinates..], output_left, t);
                }
                if has_right {
                    let coordinates = self.signs.coordinates(true, idx);
                    xor_bits(&self.signs.1[coordinates..], output_right, t);
                }
            });
        correct_node
    }
    fn correct_single(
        &self,
        signs: impl Iterator<Item = bool>,
        direction: bool,
        output: &mut Signs,
    ) -> Node {
        let mut correct_node = Node::default();
        let t = output.1;
        signs
            .enumerate()
            .take(self.seeds.len())
            .for_each(|(idx, bit)| {
                if !bit {
                    return;
                }
                correct_node ^= self.seeds[idx];
                let coordinates = self.signs.coordinates(direction, idx);
                xor_bits(&self.signs.1[coordinates..], &mut output.0, t);
            });
        correct_node
    }
}

fn xor_bits(src: &[Node], dest: &mut [Node], bit_count: usize) {
    let to_move = (bit_count + 127) >> 7;
    for i in 0..to_move {
        dest[i] ^= src[i];
    }
    // src[to_move - 1] ^= dest[to_move - 1];
}
fn copy_bits(src: &[Node], dest: &mut [Node], bit_count: usize) {
    let to_move = (bit_count + 127) >> 7;
    for i in 0..to_move {
        dest[i] = src[i];
    }
    // src[to_move - 1] ^= dest[to_move - 1];
}
// fn move_bits(src: &[Node], dest: &mut [Node], bit_count: usize, xor: bool) {
//     let mut bits_to_take_remaining = bit_count;
//     let mut current_source_bits_pointer = src_start;
//     let mut current_dest_bits_pointer = dest_start;
//     // handle remaining nodes
//     while bits_to_take_remaining > 0 {
//         let current_bit_take_offset = current_source_bits_pointer & 127;
//         let current_bit_put_offset = current_dest_bits_pointer & 127;
//         let bits_to_take_now = usize::min(
//             usize::min(128 - current_bit_take_offset, 128 - current_bit_put_offset),
//             bits_to_take_remaining,
//         );

//         let take_from_idx = current_source_bits_pointer >> 7;
//         let put_into_idx = current_dest_bits_pointer >> 7;

//         let mut bits_taken = src[take_from_idx];
//         bits_taken.shr(current_bit_take_offset as u32);
//         bits_taken.mask_lsbs(bits_to_take_now);

//         bits_taken.shl(current_bit_put_offset as u32);
//         // This is since we don't assume dest is zeroed.
//         if !xor {
//             dest[put_into_idx].mask_bits_lsbs(
//                 current_bit_put_offset,
//                 current_bit_put_offset + bits_to_take_now,
//             );
//         }
//         dest[put_into_idx] ^= bits_taken;

//         // Update pointers
//         current_source_bits_pointer += bits_to_take_now;
//         current_dest_bits_pointer += bits_to_take_now;
//         bits_to_take_remaining -= bits_to_take_now;
//     }
// }
// fn xor_bits(src: &[Node], dest: &mut [Node], bit_count: usize) {
//     move_bits(src, dest, bit_count, true)
// }
// fn copy_bits(src: &[Node], dest: &mut [Node], bit_count: usize) {
//     move_bits(src, dest, bit_count, false)
// }

// This is used to store the signs of the CWs.
// Each CW is of length t*(Lambda + 2*t) bits.
// This struct stores the 2*t*t sign bits.
#[derive(Debug, Clone)]
pub struct SignsCW(usize, Vec<Node>, usize);
impl SignsCW {
    pub fn new(t: usize, depth: usize, mut rng: impl RngCore + CryptoRng) -> Self {
        // At depth 'depth' there are at most 1<<depth non-zero paths.
        let min = usize::min(t, 1 << depth);
        let nodes_per_point_per_direction = t.div_ceil(128);
        let total_nodes = 2 * min * nodes_per_point_per_direction;
        let v = (0..total_nodes).map(|_| Node::random(&mut rng)).collect();
        Self(t, v, nodes_per_point_per_direction)
    }

    fn coordinates(&self, direction: bool, point_idx: usize) -> usize {
        let nodes_per_point_per_direction = self.2;
        // let nodes_per_point = 2 * nodes_per_point_per_direction;
        ((point_idx << 1) + (direction as usize)) * nodes_per_point_per_direction
    }

    pub fn get_sign(&self, input_sign: &mut Signs, direction: bool, point_idx: usize) {
        let t = self.0;
        let src_start = self.coordinates(direction, point_idx);
        let bit_count = t;
        copy_bits(&self.1[src_start..], &mut input_sign.0, bit_count)
    }

    pub fn put_sign(&mut self, input_sign: &Signs, direction: bool, point_idx: usize) {
        let t = self.0;
        let dest_start = self.coordinates(direction, point_idx);
        let bit_count = t;
        copy_bits(&input_sign.0, &mut self.1[dest_start..], bit_count)
    }
    pub fn flip_bit(&mut self, direction: bool, point_idx: usize, bit_idx: usize) {
        let coordinates = self.coordinates(direction, point_idx);
        let node_idx = coordinates + (bit_idx / 128);
        let in_node_idx = bit_idx & 127;
        let xor_node = Node::from(1u128 << in_node_idx);
        self.1[node_idx] ^= xor_node;
    }
}

// This is used merely for PRG expansion. A t-bit long array.
#[derive(Debug)]
pub struct Signs(Vec<Node>, usize);

impl Signs {
    pub fn new(t: usize) -> Self {
        let total_nodes = (t + 127) >> 7;
        Self(vec![Node::default(); total_nodes], t)
    }
    pub fn zero(&mut self) {
        self.0.iter_mut().for_each(|v| *v = Node::default());
    }
    pub fn fill_with_seed(&mut self, seed: &Node, direction: usize) {
        let children_low = (2 + self.0.len() * direction) as u16;
        let children_high = children_low + self.0.len() as u16;
        many_prg(seed, children_low..children_high, &mut self.0);
    }
    pub fn xor_into(&mut self, a: &Self, b: &Self) {
        self.0
            .iter_mut()
            .zip(a.0.iter().zip(b.0.iter()))
            .for_each(|(di, (ai, bi))| *di = ai ^ bi);
    }
    pub fn set_bit(&mut self, bit_idx: usize, value: bool) {
        let node_idx = bit_idx >> 7;
        let in_node_idx = bit_idx & 127;
        self.0[node_idx].set_bit_lsb(in_node_idx, value);
    }
    pub fn get_bit(&mut self, bit_idx: usize) -> bool {
        let node_idx = bit_idx >> 7;
        let in_node_idx = bit_idx & 127;
        self.0[node_idx].get_bit_lsb(in_node_idx)
    }
    fn iter(&self) -> SignsIter<'_> {
        SignsIter::new(&self.0, 0, self.1)
    }
}
struct SignsIter<'a> {
    src: &'a [Node],
    cur_idx: usize,
    dst_idx: usize,
}
impl<'a> SignsIter<'a> {
    fn new(src: &'a [Node], start: usize, count: usize) -> Self {
        Self {
            src,
            cur_idx: start,
            dst_idx: start + count,
        }
    }
}
impl<'a> Iterator for SignsIter<'a> {
    type Item = bool;
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_idx == self.dst_idx {
            return None;
        }
        let idx = self.cur_idx >> 7;
        let in_node_idx = self.cur_idx & 127;
        self.cur_idx += 1;
        Some(self.src[idx].get_bit_lsb(in_node_idx))
    }
}

// This is used merely to maintain the state during PRG.
// This is essentially a t x t boolean array.
#[derive(Clone)]
pub struct KeyGenSigns(usize, Vec<Node>, usize);
impl KeyGenSigns {
    pub fn new(t: usize) -> Self {
        let nodes_per_point = t.div_ceil(128);
        let total_nodes = t * nodes_per_point;
        let signs = vec![Node::default(); total_nodes];
        Self(t, signs, nodes_per_point)
    }
    fn set_bit(&mut self, point: usize, bit_idx: usize, value: bool) {
        let node_idx = self.coordinates(point) + (bit_idx >> 7);
        let in_node_idx = bit_idx & 127;
        self.1[node_idx].set_bit_lsb(in_node_idx, value)
    }
    fn get_bit(&self, point: usize, bit_idx: usize) -> bool {
        let node_idx = self.coordinates(point) + (bit_idx >> 7);
        let in_node_idx = bit_idx & 127;
        self.1[node_idx].get_bit_lsb(in_node_idx)
    }
    fn set_signs(&mut self, point: usize, signs: &Signs) {
        let node_idx = self.coordinates(point);
        copy_bits(&signs.0, &mut self.1[node_idx..], self.0);
    }
    fn coordinates(&self, point: usize) -> usize {
        self.2 * point
    }
    fn iter(&self, point: usize) -> SignsIter {
        let coords = self.coordinates(point);
        SignsIter::new(&self.1[coords..], 0, self.0)
    }
}

pub struct BigStateDmpfKey<Output: DpfOutput> {
    sign: bool,
    root: Node,
    cws: Vec<CW>,
    conv_cw: ConvCW<Output>,
}

fn u128_get_bit(v: u128, bit_idx: usize) -> bool {
    (v >> (127 - bit_idx)) & 1 == 1
}

pub struct BigStateSession {
    sign: Signs,
    new_sign: Signs,
}

impl DmpfSession for BigStateSession {
    fn get_session(kvs_count: usize) -> Self {
        Self {
            sign: Signs::new(kvs_count),
            new_sign: Signs::new(kvs_count),
        }
    }
}

impl<Output: DpfOutput> DmpfKey<Output> for BigStateDmpfKey<Output> {
    type Session = BigStateSession;
    fn point_count(&self) -> usize {
        self.conv_cw.0.len()
    }
    fn eval_with_session(&self, input: &u128, output: &mut Output, session: &mut Self::Session) {
        let mut seed = self.root;
        session.sign.zero();
        session.sign.set_bit(0, self.sign);
        let input_len = self.cws.len();
        let mut sign_ref = unsafe { (&mut session.sign as *mut Signs).as_mut().unwrap() };
        let mut next_sign_ref = unsafe { (&mut session.new_sign as *mut Signs).as_mut().unwrap() };
        for depth in 0..input_len {
            let d = double_prg(&seed, &DOUBLE_PRG_CHILDREN);
            let input_i = u128_get_bit(*input, depth);
            let input_i_usize = input_i as usize;
            let next_seed = d[input_i_usize];
            next_sign_ref.fill_with_seed(&seed, input_i_usize);
            seed = self.cws[depth].correct_single(sign_ref.iter(), input_i, next_sign_ref);
            seed ^= next_seed;
            (next_sign_ref, sign_ref) = (sign_ref, next_sign_ref);
        }
        *output = Output::from(seed) + self.conv_cw.conv_correct(&sign_ref);
        if self.sign {
            *output = output.neg();
        }
    }

    fn eval_all_with_session(&self, session: &mut Self::Session) -> Vec<Output> {
        let input_len = self.cws.len();
        let session_left = unsafe { (&mut session.sign as *mut Signs).as_mut().unwrap() };
        // let session_right = unsafe { (&mut session.new_sign as *mut Signs).as_mut().unwrap() };
        let mut eval_all_state = EvalAllState::new(self.point_count(), input_len);
        let mut next_eval_all_state = EvalAllState::new(self.point_count(), input_len);
        let mut seeds = Vec::with_capacity(1 << input_len);
        let mut next_seeds = Vec::with_capacity(1 << input_len);
        unsafe { seeds.set_len(1 << input_len) };
        unsafe { next_seeds.set_len(1 << input_len) };
        seeds[0] = self.root;
        session_left.zero();
        session_left.set_bit(0, self.sign);
        eval_all_state.put_sign(0, &session_left);
        for depth in 0..input_len {
            double_prg_many(
                &seeds[..1 << depth],
                &DOUBLE_PRG_CHILDREN,
                &mut next_seeds[..1 << (depth + 1)],
            );
            next_eval_all_state.fill_from_seeds(&seeds[..1 << depth]);
            (seeds, next_seeds) = (next_seeds, seeds);
            let cur_cw = &self.cws[depth];
            for path_idx in 0..1 << depth {
                let session_left =
                    unsafe { next_eval_all_state.get_sign_mut_after_expansion(path_idx, false) };
                let session_right =
                    unsafe { next_eval_all_state.get_sign_mut_after_expansion(path_idx, true) };
                let corrected_node = cur_cw.correct_bits(
                    eval_all_state.iter_sign(path_idx),
                    true,
                    true,
                    session_left,
                    session_right,
                    self.point_count(),
                );
                seeds[2 * path_idx] ^= corrected_node;
                seeds[2 * path_idx + 1] ^= corrected_node;
            }
            (next_eval_all_state, eval_all_state) = (eval_all_state, next_eval_all_state)
        }
        seeds
            .into_iter()
            .enumerate()
            .map(|(idx, seed)| {
                eval_all_state.get_sign(idx, &mut session.sign);
                let o = Output::from(seed) + self.conv_cw.conv_correct(&session.sign);
                if self.sign {
                    -o
                } else {
                    o
                }
            })
            .collect()
    }
}
struct EvalAllState {
    t: usize,
    signs: Vec<Node>,
    nodes_per_path: usize,
}
impl EvalAllState {
    fn new(points: usize, input_len: usize) -> Self {
        let output_size = 1 << input_len;
        // multiply by 2 because there are two directions.
        let nodes_per_path_after_expand = points.div_ceil(128) * 2;
        let total_nodes = nodes_per_path_after_expand * output_size;
        let mut signs = Vec::with_capacity(total_nodes);
        unsafe { signs.set_len(total_nodes) };
        Self {
            t: points,
            signs,
            nodes_per_path: points.div_ceil(128),
        }
    }
    fn fill_from_seeds(&mut self, seeds: &[Node]) {
        let nodes_per_path_after_expand = self.nodes_per_path * 2;
        let nodes_per_path_u16 = nodes_per_path_after_expand as u16;
        let output_len = seeds.len() * nodes_per_path_after_expand;
        many_many_prg(
            seeds,
            2..2 + nodes_per_path_u16,
            &mut self.signs[..output_len],
        )
    }
    fn coordinates(&self, point: usize) -> usize {
        point * self.nodes_per_path
    }
    fn put_sign(&mut self, point: usize, sign: &Signs) {
        let dst = self.coordinates(point);
        copy_bits(&sign.0, &mut self.signs[dst..], self.t);
    }
    fn get_sign(&self, point: usize, sign: &mut Signs) {
        let src = self.coordinates(point);
        copy_bits(&self.signs[src..], &mut sign.0, self.t);
    }
    unsafe fn get_sign_mut_after_expansion(&self, point: usize, direction: bool) -> &mut [Node] {
        let nodes_per_path_per_direction = self.nodes_per_path;
        let nodes_per_path_after_expand = nodes_per_path_per_direction * 2;
        let start = (nodes_per_path_after_expand * point)
            + (nodes_per_path_per_direction * (direction as usize));
        let mutable_self = unsafe { (self as *const Self as *mut Self).as_mut().unwrap() };
        &mut mutable_self.signs[start..start + nodes_per_path_per_direction]
    }
    fn iter_sign(&self, point: usize) -> SignsIter {
        let start = self.coordinates(point);
        SignsIter::new(&self.signs[start..], 0, self.t)
    }
}

#[cfg(test)]
mod tests {
    use super::{Signs, SignsCW};
    use crate::{big_state::BigStateDmpf, rb_okvs::OkvsValue, Dmpf, DmpfKey, PrimeField64x2};
    use rand::{thread_rng, RngCore};
    use std::collections::HashMap;

    fn encode_input(v: u128, input_len: usize) -> u128 {
        v << (128 - input_len)
    }
    #[test]
    fn test() {
        const INPUT_LEN: usize = 11;
        const INPUTS: usize = 439;
        assert!(INPUTS <= 1 << (INPUT_LEN - 1));
        let mut rng = thread_rng();
        let mut inputs_hashmap = HashMap::with_capacity(INPUTS);
        while inputs_hashmap.len() < INPUTS {
            let v = ((rng.next_u64() % (1 << INPUT_LEN)) as u128) << (128 - INPUT_LEN);
            inputs_hashmap.insert(v, PrimeField64x2::random(&mut rng));
        }
        let mut kvs: Vec<_> = inputs_hashmap.iter().map(|(k, v)| (*k, *v)).collect();
        kvs.sort_by_cached_key(|v| v.0);

        let dmpf = BigStateDmpf::new();
        let (k_0, k_1) = dmpf.try_gen(INPUT_LEN, &kvs, &mut rng).unwrap();
        let eval_all_0 = k_0.eval_all();
        let eval_all_1 = k_1.eval_all();
        let eval_all_sum: Vec<_> = eval_all_0
            .iter()
            .zip(eval_all_1.iter())
            .map(|(a, b)| *a + *b)
            .collect();

        for i in 0..1 << INPUT_LEN {
            let encoded_i = encode_input(i, INPUT_LEN);
            let mut output_0 = PrimeField64x2::default();
            k_0.eval(&encoded_i, &mut output_0);
            let mut output_1 = PrimeField64x2::default();
            k_1.eval(&encoded_i, &mut output_1);
            assert_eq!(output_0, eval_all_0[i as usize]);
            assert_eq!(output_1, eval_all_1[i as usize]);
            let output = output_0 + output_1;
            if inputs_hashmap.contains_key(&encoded_i) {
                assert_eq!(output, inputs_hashmap[&encoded_i])
            } else {
                assert!(output.is_zero())
            }
        }
    }
    #[test]
    fn test_sign() {
        let t = 1000;
        let mut sign = Signs::new(t);
        let mut sign_cw = SignsCW::new(t, 10, thread_rng());
        for i in 0..t {
            sign.set_bit(i, thread_rng().next_u64() & 1 == 1);
        }
        sign_cw.put_sign(&sign, true, t / 2);
        let mut new_sign = Signs::new(t);
        sign_cw.get_sign(&mut new_sign, true, t / 2);
        for i in 0..t {
            assert_eq!(sign.get_bit(i), new_sign.get_bit(i));
        }
    }
}
