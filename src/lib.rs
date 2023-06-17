use std::ops::BitXorAssign;

pub mod big_state;
mod dpf;
mod trie;
pub use dpf::int_to_bits;

pub use dpf::BitVec;
pub use dpf::DpfKey;
pub use dpf::Node;
pub const BITS_IN_BYTE: usize = 8;
pub const BITS_OF_SECURITY: usize = 128;
pub const BYTES_OF_SECURITY: usize = BITS_OF_SECURITY / BITS_IN_BYTE;

fn xor_arrays<const LENGTH: usize, T: Copy + BitXorAssign>(
    lhs: &mut [T; LENGTH],
    rhs: &[T; LENGTH],
) {
    for i in 0..LENGTH {
        lhs[i] ^= rhs[i];
    }
}
