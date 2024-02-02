use crate::{utils::Node, xor_arrays, BYTES_OF_SECURITY};
use aes::{
    cipher::{BlockEncrypt, KeyInit},
    Aes128, Block,
};
use once_cell::sync::Lazy;
const DPF_AES_KEY: [u8; BYTES_OF_SECURITY] =
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2];
static AES: Lazy<Aes128> = Lazy::new(|| Aes128::new_from_slice(&DPF_AES_KEY).unwrap());

pub(crate) const DOUBLE_PRG_CHILDREN: [u8; 2] = [0, 1];

pub fn double_prg(input: &Node, children: &[u8; 2]) -> [Node; 2] {
    let mut blocks = [Block::from(*<Node as AsRef<[u8; 16]>>::as_ref(input)); 2];
    let blocks_node = unsafe { (blocks.as_mut_ptr() as *mut [Node; 2]).as_mut().unwrap() };
    blocks[0][0] ^= children[0];
    blocks[1][0] ^= children[1];
    AES.encrypt_blocks(&mut blocks);
    blocks_node[0] ^= input;
    blocks_node[1] ^= input;
    blocks[0][0] ^= children[0];
    blocks[1][0] ^= children[1];
    unsafe { std::mem::transmute(blocks) }
}
pub fn double_prg_many(input: &[Node], children: &[u8; 2], output: &mut [Node]) {
    const BLOCK_SIZE_INPUT: usize = 1;
    // The less interesting case
    if input.len() < BLOCK_SIZE_INPUT {
        input
            .iter()
            .zip(output.chunks_exact_mut(2))
            .for_each(|(i, o)| {
                let d = double_prg(i, children);
                (o[0], o[1]) = (d[0], d[1]);
            })
    };

    input
        .chunks_exact(BLOCK_SIZE_INPUT)
        .zip(output.chunks_exact_mut(2 * BLOCK_SIZE_INPUT))
        .for_each(|(i, o)| {
            for idx in 0..BLOCK_SIZE_INPUT {
                o[2 * idx] = i[idx];
                o[2 * idx + 1] = i[idx];
            }
            let o_u8 = unsafe {
                std::slice::from_raw_parts_mut(o.as_mut_ptr() as *mut Block, 2 * BLOCK_SIZE_INPUT)
            };
            for idx in 0..BLOCK_SIZE_INPUT {
                o_u8[2 * idx][0] ^= children[0];
                o_u8[2 * idx + 1][0] ^= children[1];
            }
            AES.encrypt_blocks(o_u8);
            for idx in 0..BLOCK_SIZE_INPUT {
                o[2 * idx] ^= i[idx];
                o[2 * idx + 1] ^= i[idx];
            }
            for idx in 0..BLOCK_SIZE_INPUT {
                o_u8[2 * idx][0] ^= children[0];
                o_u8[2 * idx + 1][0] ^= children[1];
            }
        });
    // blocks[0][0] ^= children[0];
    // blocks[1][0] ^= children[1];
    // AES.encrypt_blocks(&mut blocks);
    // xor_arrays(&mut blocks[0].into(), input.as_ref());
    // xor_arrays(&mut blocks[1].into(), input.as_ref());
    // blocks[0][0] ^= children[0];
    // blocks[1][0] ^= children[1];
    // unsafe { std::mem::transmute(blocks) }
}
pub fn triple_prg(input: &Node, children: &[u8; 3]) -> [Node; 3] {
    let mut blocks = [Block::from(*<Node as AsRef<[u8; 16]>>::as_ref(input)); 3];
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
    let input_block = Block::from(*<Node as AsRef<[u8; 16]>>::as_ref(input));
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

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::Node;

    use super::{double_prg, double_prg_many, DOUBLE_PRG_CHILDREN};

    #[test]
    fn test_double_prg() {
        let mut rng = thread_rng();
        for log_len in 0..10 {
            let a: Vec<_> = (0..1 << log_len).map(|_| Node::random(&mut rng)).collect();
            let mut b: Vec<_> = vec![Node::default(); 1 << (log_len + 1)];
            double_prg_many(&a, &DOUBLE_PRG_CHILDREN, &mut b);
            a.iter().zip(b.chunks_exact(2)).for_each(|(ai, bs)| {
                let d = double_prg(ai, &DOUBLE_PRG_CHILDREN);
                assert_eq!(d[0], bs[0]);
                assert_eq!(d[1], bs[1]);
            })
        }
    }
}
