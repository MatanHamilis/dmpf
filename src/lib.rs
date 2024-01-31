use std::iter::Sum;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::BitXorAssign;
use std::ops::Neg;
use std::ops::Sub;

// pub mod big_state;
mod dpf;
pub mod field;
mod prg;
mod trie;
mod utils;
pub use dpf::int_to_bits;
pub mod batch_code;
pub mod okvs;
mod rb_okvs;

pub use dpf::DpfKey;
use rand::CryptoRng;
use rand::RngCore;
use rb_okvs::OkvsValue;
pub use utils::Node;
pub use utils::{BitSlice, BitSliceMut, BitVec};
pub const BITS_IN_BYTE: usize = 8;
pub const BITS_OF_SECURITY: usize = 128;
pub const BYTES_OF_SECURITY: usize = BITS_OF_SECURITY / BITS_IN_BYTE;
pub use dpf::DpfDmpf;
pub use field::{PrimeField64, PrimeField64x2, RadixTwoFftFriendFieldElement};
pub use rb_okvs::EpsilonPercent;
pub use utils::SmallFieldContainer;

pub trait DpfOutput:
    Sub<Output = Self>
    + Add<Output = Self>
    + From<Node>
    + Sum
    + Default
    + Neg<Output = Self>
    + AddAssign
    + Eq
    + PartialEq
    + Copy
    + Clone
    + OkvsValue
{
}

fn xor_arrays<const LENGTH: usize, T: Copy + BitXorAssign>(
    lhs: &mut [T; LENGTH],
    rhs: &[T; LENGTH],
) {
    for i in 0..LENGTH {
        lhs[i] ^= rhs[i];
    }
}

pub trait Dmpf<Output: DpfOutput>
where
    Self: Sized,
{
    type Key: DmpfKey<Output>;
    fn try_gen<R: CryptoRng + RngCore>(
        &self,
        input_length: usize,
        inputs: &[(u128, Output)],
        rng: &mut R,
    ) -> Option<(Self::Key, Self::Key)>;
}

pub trait DmpfSession {
    fn get_session(kvs_count: usize) -> Self;
}
pub struct EmptySession;
impl DmpfSession for EmptySession {
    fn get_session(kvs_count: usize) -> Self {
        Self
    }
}

pub trait DmpfKey<Output>
where
    Self: Sized,
    Output: DpfOutput,
{
    type Session: DmpfSession;
    fn make_session(&self) -> Self::Session;
    fn eval(&self, input: &u128, output: &mut Output) {
        self.eval_with_session(input, output, self.make_session())
    }
    fn eval_with_session(&self, input: &u128, output: &mut Output, session: Self::Session);
    fn eval_all(&self) -> Vec<Output> {
        self.eval_all_with_session(self.make_session())
    }
    fn eval_all_with_session(&self, session: Self::Session) -> Vec<Output>;
}

pub(crate) fn random_u128<R: CryptoRng + RngCore>(rng: &mut R) -> u128 {
    ((rng.next_u64() as u128) << 64) ^ (rng.next_u64() as u128)
}
pub(crate) fn random_u126<R: CryptoRng + RngCore>(rng: &mut R) -> u128 {
    random_u128(rng) & (!3u128)
}
