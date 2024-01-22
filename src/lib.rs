use std::ops::BitXorAssign;

pub mod big_state;
mod dpf;
mod prg;
mod trie;
pub use dpf::int_to_bits;
pub mod batch_code;
pub mod okvs;

pub use dpf::DpfKey;
pub use dpf::Node;
pub use dpf::{BitSlice, BitSliceMut, BitVec};
use rand::CryptoRng;
use rand::RngCore;
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

pub trait Dmpf
where
    Self: Sized,
{
    type Key: DmpfKey;
    fn try_gen<R: CryptoRng + RngCore>(
        &self,
        inputs: &[(
            <Self::Key as DmpfKey>::InputContainer,
            <Self::Key as DmpfKey>::OutputContainer,
        )],
        rng: &mut R,
    ) -> Option<(Self::Key, Self::Key)>;
}

pub trait DmpfKey
where
    Self: Sized,
{
    type Session;
    type InputContainer;
    type OutputContainer;
    fn make_session(&self) -> Self;
    fn eval(&self, input: &Self::InputContainer, output: &mut Self::OutputContainer);
    fn eval_all(&self) -> Box<[Self::OutputContainer]>;
}

pub(crate) fn random_u128<R: CryptoRng + RngCore>(rng: &mut R) -> u128 {
    ((rng.next_u64() as u128) << 64) ^ (rng.next_u64() as u128)
}
pub(crate) fn random_u126<R: CryptoRng + RngCore>(rng: &mut R) -> u128 {
    random_u128(rng) & (!3u128)
}
