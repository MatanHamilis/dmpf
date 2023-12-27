use std::ops::BitXorAssign;

// mod batch_code;
pub mod big_state;
mod dpf;
mod prg;
mod trie;
pub use dpf::int_to_bits;
pub mod okvs;

pub use dpf::BitVec;
pub use dpf::DpfKey;
pub use dpf::Node;
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
