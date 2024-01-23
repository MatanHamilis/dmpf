//! This module implements polynomials ring

use std::{
    marker::PhantomData,
    ops::{AddAssign, Mul},
};

use crate::{
    field::RadixTwoFftFriendFieldElement,
    polynomial::{DensePolynomial, SparsePolynomial},
};

pub trait ModuloPolynomial<F: RadixTwoFftFriendFieldElement>: Clone {
    fn modulo(&self, p: DensePolynomial<F>) -> DensePolynomial<F>;
}
#[derive(Clone)]
pub struct TwoPowerDegreeCyclotomicPolynomial<F: RadixTwoFftFriendFieldElement> {
    log_n: usize,
    p: PhantomData<F>,
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
            coeffs[i & deg_mask] += c;
        }
        coeffs.resize(output_degree, F::zero());
        p
    }
}

pub struct PolynomialRingElement<F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>> {
    p: DensePolynomial<F>,
    modulo: M,
}
impl<F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>> PolynomialRingElement<F, M> {
    pub fn new(p: DensePolynomial<F>, modulo: M) -> Self {
        Self { p, modulo }
    }
}

impl<'a, F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>> Mul
    for &PolynomialRingElement<F, M>
{
    type Output = PolynomialRingElement<F, M>;
    fn mul(self, rhs: Self) -> Self::Output {
        let p = &self.p * &rhs.p;
        let o = self.modulo.modulo(p);
        PolynomialRingElement {
            p: o,
            modulo: self.modulo,
        }
    }
}
impl<'a, F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>> AddAssign
    for PolynomialRingElement<F, M>
{
    fn add_assign(&mut self, rhs: Self) {
        self.p.add_assign(rhs.p);
    }
}

pub struct SparsePolynomialRingElement<F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>> {
    p: SparsePolynomial<F>,
    modulo: M,
}
impl<F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>> SparsePolynomialRingElement<F, M> {
    pub fn new(p: SparsePolynomial<F>, modulo: M) -> Self {
        Self { p, modulo }
    }
}

impl<'a, F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>> Mul<&PolynomialRingElement<F, M>>
    for &SparsePolynomialRingElement<F, M>
{
    type Output = PolynomialRingElement<F, M>;
    fn mul(self, rhs: &PolynomialRingElement<F, M>) -> Self::Output {
        let p = &rhs.p * &self.p;
        PolynomialRingElement {
            p,
            modulo: self.modulo,
        }
    }
}

impl<'a, F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>> Mul
    for &SparsePolynomialRingElement<F, M>
{
    type Output = SparsePolynomialRingElement<F, M>;
    fn mul(self, rhs: &SparsePolynomialRingElement<F, M>) -> Self::Output {
        let p = &rhs.p * &self.p;
        SparsePolynomialRingElement {
            p,
            modulo: self.modulo,
        }
    }
}
