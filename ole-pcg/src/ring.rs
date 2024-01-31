//! This module implements polynomials ring

use std::{
    iter::Sum,
    marker::PhantomData,
    ops::{Add, AddAssign, Mul},
};

use crate::polynomial::{DensePolynomial, SparsePolynomial};
use dmpf::field::RadixTwoFftFriendFieldElement;
use rand::{CryptoRng, RngCore};

pub trait ModuloPolynomial<F: RadixTwoFftFriendFieldElement>: Clone {
    fn modulo(&self, p: DensePolynomial<F>) -> DensePolynomial<F>;
    fn deg(&self) -> usize;
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwoPowerDegreeCyclotomicPolynomial<F: RadixTwoFftFriendFieldElement> {
    log_n: usize,
    p: PhantomData<F>,
}
impl<F: RadixTwoFftFriendFieldElement> TwoPowerDegreeCyclotomicPolynomial<F> {
    pub fn new(log_n: usize) -> Self {
        Self {
            log_n,
            p: PhantomData,
        }
    }
}
impl<F: RadixTwoFftFriendFieldElement> ModuloPolynomial<F>
    for TwoPowerDegreeCyclotomicPolynomial<F>
{
    fn modulo(&self, mut p: DensePolynomial<F>) -> DensePolynomial<F> {
        let output_degree = 1 << self.log_n;
        let deg_mask = output_degree - 1;
        let coeffs = &mut p.coefficients;
        for i in output_degree..coeffs.len() {
            let c = coeffs[i];
            coeffs[i & deg_mask] -= c;
        }
        coeffs.resize(output_degree, F::zero());
        p
    }
    fn deg(&self) -> usize {
        1 << self.log_n
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolynomialRingElement<F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>> {
    p: DensePolynomial<F>,
    modulo: M,
}
impl<F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>> PolynomialRingElement<F, M> {
    pub fn new(p: DensePolynomial<F>, modulo: M) -> Self {
        Self { p, modulo }
    }
    pub fn get_modulo(&self) -> &M {
        &self.modulo
    }
    pub fn get_coefficients(&self) -> &DensePolynomial<F> {
        &self.p
    }
    pub fn random<R: RngCore + CryptoRng>(rng: R, modulo: M) -> Self {
        Self {
            p: DensePolynomial::random(rng, modulo.deg()),
            modulo,
        }
    }
}

impl<'a, F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>> Mul
    for &PolynomialRingElement<F, M>
{
    type Output = PolynomialRingElement<F, M>;
    fn mul(self, mut rhs: Self) -> Self::Output {
        let p = &self.p * &rhs.p;
        let o = self.modulo.modulo(p);
        PolynomialRingElement {
            p: o,
            modulo: self.modulo.clone(),
        }
    }
}
impl<'a, F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>>
    AddAssign<&PolynomialRingElement<F, M>> for PolynomialRingElement<F, M>
{
    fn add_assign(&mut self, rhs: &Self) {
        self.p.add_assign(&rhs.p);
    }
}
impl<'a, F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>> Add
    for &PolynomialRingElement<F, M>
{
    type Output = PolynomialRingElement<F, M>;
    fn add(self, rhs: Self) -> Self::Output {
        let mut s = self.clone();
        s.add_assign(&rhs);
        s
    }
}

pub struct SparsePolynomialRingElement<F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>> {
    pub(crate) p: SparsePolynomial<F>,
    modulo: M,
}
impl<F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>> SparsePolynomialRingElement<F, M> {
    pub fn new(p: SparsePolynomial<F>, modulo: M) -> Self {
        Self { p, modulo }
    }
    pub fn modulo(&self) -> &M {
        &self.modulo
    }
    pub fn random<R: CryptoRng + RngCore>(rng: R, modulo: M, weight: usize) -> Self {
        SparsePolynomialRingElement::new(
            SparsePolynomial::random(rng, modulo.deg(), weight),
            modulo,
        )
    }
}

impl<'a, F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>> Mul<&PolynomialRingElement<F, M>>
    for &SparsePolynomialRingElement<F, M>
{
    type Output = PolynomialRingElement<F, M>;
    fn mul(self, rhs: &PolynomialRingElement<F, M>) -> Self::Output {
        let p_mul = &rhs.p * &self.p;
        let p = self.modulo.modulo(p_mul);
        PolynomialRingElement {
            p,
            modulo: self.modulo.clone(),
        }
    }
}

// impl<'a, F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>> Mul
//     for &SparsePolynomialRingElement<F, M>
// {
//     type Output = SparsePolynomialRingElement<F, M>;
//     fn mul(self, rhs: &SparsePolynomialRingElement<F, M>) -> Self::Output {
//         let p = self.modulo.modulo(&rhs.p * &self.p);

//         SparsePolynomialRingElement {
//             p,
//             modulo: self.modulo.clone(),
//         }
//     }
// }

impl<F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>> Sum for PolynomialRingElement<F, M> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let mut output = iter.next().unwrap();
        iter.for_each(|v| output += &v);
        output
    }
}

#[cfg(test)]
mod tests {
    use std::iter::Cycle;

    use dmpf::PrimeField64;
    use rand::thread_rng;

    use super::{PolynomialRingElement, TwoPowerDegreeCyclotomicPolynomial};

    #[test]
    fn test() {
        const LOG_DEG: usize = 10;
        let modulo = TwoPowerDegreeCyclotomicPolynomial::<PrimeField64>::new(LOG_DEG);
        let a = PolynomialRingElement::random(thread_rng(), modulo.clone());
        let b = PolynomialRingElement::random(thread_rng(), modulo);
        let bb = &b * &b;
        let ab = &a * &b;
        let ba = &a * &b;

        let mut a_clone = a.clone();
        a_clone += &b;
        assert_eq!(&a * &b, &b * &a);
        assert_eq!(&ab * &b, &a * &bb);
    }
}
