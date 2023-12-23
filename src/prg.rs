use crate::{dpf::Node, xor_arrays, BYTES_OF_SECURITY};
use aes::{
    cipher::{BlockEncrypt, KeyInit},
    Aes128, Block,
};
use once_cell::sync::Lazy;
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
