use std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::rb_okvs::OkvsValue;
use rand::{CryptoRng, RngCore};

use crate::{DpfOutput, Node};

pub trait FieldElement:
    Add<Output = Self>
    + Sub<Output = Self>
    + AddAssign
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div
    + DivAssign
    + Clone
    + Copy
    + Sum
    + PartialEq
    + Neg
    + Eq
{
    fn zero() -> Self;
    fn one() -> Self;
    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    fn inv(self) -> Self;
    fn square(&self) -> Self {
        *self * *self
    }
    fn random<R: CryptoRng + RngCore>(rng: R) -> Self;
}

pub trait RadixTwoFftFriendFieldElement: FieldElement + From<usize> {
    fn generator_for_fft(log_fft_size: usize) -> Self;
}

const CHUNK_SIZE: usize = 16;
pub struct PowersIterator<F: FieldElement> {
    element_power: F,
    next_powers: [F; CHUNK_SIZE],
    index: usize,
}
impl From<usize> for PrimeField64 {
    fn from(value: usize) -> Self {
        Self(value as u64) + Self::zero()
    }
}

impl<F: FieldElement> PowersIterator<F> {
    pub fn new(x: F) -> Self {
        let mut element_power = F::one();
        let next_powers = core::array::from_fn(|_| {
            let output = element_power;
            element_power *= x;
            output
        });
        Self {
            element_power,
            next_powers,
            index: 0,
        }
    }
}

impl<F: FieldElement> Iterator for PowersIterator<F> {
    type Item = F;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == CHUNK_SIZE {
            self.index = 0;
            self.next_powers
                .iter_mut()
                .for_each(|v| *v *= self.element_power)
        };
        let output = self.next_powers[self.index];
        self.index += 1;
        Some(output)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrimeField64(u64);
impl PrimeField64 {
    const OVERFLOW: u64 = (1 << 32) - 1;
    const MOD: u64 = 0xFFFFFFFF00000001;
    const MOD_LARGE: u128 = Self::MOD as u128;
    fn repeated_square<const N: usize>(&self) -> Self {
        let mut output = *self;
        for _ in 0..N {
            output = output.square();
        }
        output
    }
}
impl Default for PrimeField64 {
    fn default() -> Self {
        Self::zero()
    }
}
impl RadixTwoFftFriendFieldElement for PrimeField64 {
    fn generator_for_fft(log_fft_size: usize) -> Self {
        // This is a generator of order 2**32.
        const BASE_GENERATOR: u64 = 1_753_635_133_440_165_772;
        let mut output = Self(BASE_GENERATOR);
        for _ in 0..(32 - log_fft_size) {
            output = output.square();
        }
        output
    }
}
impl From<u64> for PrimeField64 {
    fn from(value: u64) -> Self {
        Self(value) + Self::zero()
    }
}
impl FieldElement for PrimeField64 {
    fn random<R: CryptoRng + RngCore>(mut rng: R) -> Self {
        Self(rng.next_u64()) + Self::zero()
    }
    fn one() -> Self {
        Self(1)
    }
    fn is_one(&self) -> bool {
        self.0 == 1
    }
    fn zero() -> Self {
        Self(0)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
    // Based on code from plonky2 library.
    fn inv(self) -> Self {
        assert!(!self.is_zero());
        let t2 = self.square() * self;
        let t3 = t2.square() * self;
        let t6 = t3.repeated_square::<3>() * t3;
        let t12 = t6.repeated_square::<6>() * t6;
        let t24 = t12.repeated_square::<12>() * t12;
        let t30 = t24.repeated_square::<6>() * t6;
        let t31 = t30.square() * self;
        let t63 = t31.repeated_square::<32>() * t31;
        t63.square() * self
    }
}

impl Neg for PrimeField64 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(((!self.is_zero()) as u64) * (Self::MOD - self.0))
    }
}
impl SubAssign for PrimeField64 {
    fn sub_assign(&mut self, rhs: Self) {
        let (res_wrapped, borrow) = self.0.overflowing_sub(rhs.0);
        self.0 = res_wrapped - Self::OVERFLOW * (borrow as u64);
    }
}
impl Sub for PrimeField64 {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl Add for PrimeField64 {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl AddAssign for PrimeField64 {
    fn add_assign(&mut self, rhs: Self) {
        let (res_wrapped, carry) = self.0.overflowing_add(rhs.0);
        self.0 = (res_wrapped + Self::OVERFLOW * (carry as u64)) % Self::MOD;
    }
}

impl MulAssign for PrimeField64 {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = (((self.0 as u128) * (rhs.0 as u128)) % Self::MOD_LARGE) as u64;
    }
}

impl Mul for PrimeField64 {
    type Output = Self;
    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}

impl Sum for PrimeField64 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self((iter.map(|i| i.0 as u128).sum::<u128>() % Self::MOD_LARGE) as u64)
    }
}
impl DivAssign for PrimeField64 {
    fn div_assign(&mut self, rhs: Self) {
        *self *= rhs.inv();
    }
}
impl Div for PrimeField64 {
    type Output = Self;
    fn div(mut self, rhs: Self) -> Self::Output {
        self /= rhs;
        self
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct PrimeField64x2(PrimeField64, PrimeField64);
impl PrimeField64x2 {
    pub fn random<R: RngCore + CryptoRng>(mut rng: R) -> Self {
        Self(
            PrimeField64::random(&mut rng),
            PrimeField64::random(&mut rng),
        )
    }
}
impl DpfOutput for PrimeField64x2 {}
impl From<bool> for PrimeField64x2 {
    fn from(value: bool) -> Self {
        Self(PrimeField64(value as u64), PrimeField64(value as u64))
    }
}
impl Mul for PrimeField64x2 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0, self.1 * rhs.1)
    }
}
impl SubAssign for PrimeField64x2 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
        self.1 -= rhs.1;
    }
}
impl Div for PrimeField64x2 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0, self.1 / rhs.1)
    }
}
impl AddAssign for PrimeField64x2 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}
impl Mul<bool> for PrimeField64x2 {
    type Output = Self;
    fn mul(self, rhs: bool) -> Self::Output {
        Self(
            PrimeField64(self.0 .0 * (rhs as u64)),
            PrimeField64(self.1 .0 * (rhs as u64)),
        )
    }
}
impl OkvsValue for PrimeField64x2 {
    fn random<R: CryptoRng + RngCore>(rng: R) -> Self {
        Self::random(rng)
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero() || self.1.is_zero()
    }
    fn inv(&self) -> Self {
        Self(self.0.inv(), self.1.inv())
    }
}
impl MulAssign for PrimeField64x2 {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
        self.1 *= rhs.1;
    }
}
impl Neg for PrimeField64x2 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        PrimeField64x2(self.0.neg(), self.1.neg())
    }
}
impl Default for PrimeField64x2 {
    fn default() -> Self {
        Self(PrimeField64::default(), PrimeField64::default())
    }
}
impl Sum for PrimeField64x2 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |a, b| a + b)
    }
}
impl From<Node> for PrimeField64x2 {
    fn from(value: Node) -> Self {
        let value = u128::from(value);
        let lower = (value & 0xffffffffffffffffu128) as u64;
        let higher = (value >> 64) as u64;
        Self(PrimeField64::from(lower), PrimeField64::from(higher))
    }
}
impl Add for PrimeField64x2 {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}
impl Sub for PrimeField64x2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self + (rhs.neg())
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use super::{FieldElement, PrimeField64};

    #[test]
    fn test() {
        let mut rng = thread_rng();
        let a = PrimeField64::random(&mut rng);
        let b = PrimeField64::random(&mut rng);
        assert_eq!(a * b, b * a);
        assert_eq!(a * b * b.inv(), a);
        assert_eq!(a + b, b + a);
        assert_eq!(a + b - b, a);
        assert_eq!(a + b + (-b), a);
        assert_eq!(a * PrimeField64::one(), a);
        assert_eq!(a * PrimeField64::zero(), PrimeField64::zero());
    }
}
