use std::{
    borrow::Borrow,
    collections::HashMap,
    ops::{Add, AddAssign, Index, IndexMut, Mul, Sub, SubAssign},
};

use dmpf::{Dmpf, DmpfKey};

use crate::{
    fft::{fft, ifft},
    field::{PowersIterator, RadixTwoFftFriendFieldElement},
};
pub trait Polynomial<F: RadixTwoFftFriendFieldElement>:
    Add + AddAssign + SubAssign + Clone + Index<usize> + IndexMut<usize>
where
    for<'a> &'a Self: Mul<Output = Self>,
    for<'a> &'a Self: Sub<Output = Self>,
{
    fn eval(&self, input: F) -> F;
}

/// This implements a sparse polynomial represented as coefficients
pub struct SparsePolynomial<F: RadixTwoFftFriendFieldElement> {
    coefficients: Vec<(F, usize)>,
}

impl<F: RadixTwoFftFriendFieldElement> SparsePolynomial<F> {
    pub fn new(coefficients: Vec<(F, usize)>) -> Self {
        Self { coefficients }
    }
    pub fn share<D: Dmpf>(&self, points: usize, dmpf: &D) where <D::Key as DmpfKey>::InputContainer={

        let f = dmpf.try_gen(inputs, rng);
    }
}

#[derive(Clone)]
pub struct DensePolynomial<F: RadixTwoFftFriendFieldElement> {
    pub coefficients: Vec<F>,
}

impl<F: RadixTwoFftFriendFieldElement> DensePolynomial<F> {
    pub fn new(degree_bound: usize) -> Self {
        Self {
            coefficients: vec![F::zero(); degree_bound],
        }
    }
    pub fn from_iter(iter: impl Iterator<Item = F>) -> Self {
        Self {
            coefficients: Vec::from_iter(iter),
        }
    }
}

impl<F: RadixTwoFftFriendFieldElement> Polynomial<F> for DensePolynomial<F> {
    fn eval(&self, input: F) -> F {
        let powers = PowersIterator::new(input);
        self.coefficients
            .iter()
            .zip(powers)
            .map(|(&a, b)| a * b)
            .sum()
    }
}
impl<F: RadixTwoFftFriendFieldElement> Index<usize> for DensePolynomial<F> {
    type Output = F;
    fn index(&self, index: usize) -> &Self::Output {
        &self.coefficients[index]
    }
}
impl<F: RadixTwoFftFriendFieldElement> IndexMut<usize> for DensePolynomial<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coefficients[index]
    }
}
impl<F: RadixTwoFftFriendFieldElement> SubAssign for DensePolynomial<F> {
    fn sub_assign(&mut self, mut rhs: Self) {
        debug_assert_eq!(self.coefficients.len(), rhs.coefficients.len());
        self.coefficients
            .iter_mut()
            .zip(rhs.coefficients.iter())
            .for_each(|(o, i)| *o -= *i);
    }
}

impl<F: RadixTwoFftFriendFieldElement> Sub for &DensePolynomial<F> {
    type Output = DensePolynomial<F>;
    fn sub(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.coefficients.len(), rhs.coefficients.len());
        DensePolynomial::from_iter(
            self.coefficients
                .iter()
                .zip(rhs.coefficients.iter())
                .map(|(a, b)| *a - *b),
        )
    }
}

impl<F: RadixTwoFftFriendFieldElement> AddAssign for DensePolynomial<F> {
    fn add_assign(&mut self, rhs: Self) {
        debug_assert_eq!(self.coefficients.len(), rhs.coefficients.len());
        self.coefficients
            .iter_mut()
            .zip(rhs.coefficients.iter())
            .for_each(|(o, i)| *o += *i);
    }
}

impl<F: RadixTwoFftFriendFieldElement> Add for DensePolynomial<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.coefficients.len(), rhs.coefficients.len());
        Self::from_iter(
            self.coefficients
                .iter()
                .zip(rhs.coefficients.iter())
                .map(|(a, b)| *a + *b),
        )
    }
}

impl<F: RadixTwoFftFriendFieldElement> Mul for &DensePolynomial<F> {
    type Output = DensePolynomial<F>;
    fn mul(mut self, mut rhs: &DensePolynomial<F>) -> Self::Output {
        assert_eq!(self.coefficients.len(), rhs.coefficients.len());
        let log_dest_size = ((self.coefficients.len() * 2 - 1).ilog2() + 1) as usize;
        let dest_size: usize = 1 << log_dest_size;
        self.coefficients.resize(dest_size, F::zero());
        rhs.coefficients.resize(dest_size, F::zero());
        let mut self_evals = fft(&self.coefficients);
        let rhs_evals = fft(&rhs.coefficients);
        self_evals
            .iter_mut()
            .zip(rhs_evals)
            .for_each(|(a, b)| *a *= b);
        let output_coeffs = ifft(&self_evals);
        DensePolynomial {
            coefficients: output_coeffs,
        }
    }
}
impl<F: RadixTwoFftFriendFieldElement> Mul<&SparsePolynomial<F>> for &DensePolynomial<F> {
    type Output = DensePolynomial<F>;
    fn mul(self, rhs: &SparsePolynomial<F>) -> Self::Output {
        let output_len = 2 * self.coefficients.len();
        let mut output = vec![F::zero(); output_len];
        for (idx, c) in self.coefficients.into_iter().enumerate() {
            for (c_sparse, idx_sparse) in rhs.coefficients {
                output[idx + idx_sparse] += c * c_sparse;
            }
        }
        DensePolynomial {
            coefficients: output,
        }
    }
}

impl<F: RadixTwoFftFriendFieldElement> Mul for &SparsePolynomial<F> {
    type Output = SparsePolynomial<F>;
    fn mul(self, rhs: &SparsePolynomial<F>) -> Self::Output {
        let output_len = 2 * self.coefficients.len();
        let mut output = vec![F::zero(); output_len];
        let mut hash_output = HashMap::<usize, F>::new();
        for (c, idx) in self.coefficients.into_iter() {
            for (c_sparse, idx_sparse) in rhs.coefficients {
                let v = hash_output.entry(idx + idx_sparse).or_insert(F::zero());
                *v += c * c_sparse;
            }
        }
        SparsePolynomial {
            coefficients: hash_output.into_iter().map(|(a, b)| (b, a)).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::{
        fft::{fft, ifft},
        field::{FieldElement, PowersIterator, PrimeField64, RadixTwoFftFriendFieldElement},
        polynomial::Polynomial,
    };

    use super::DensePolynomial;

    #[test]
    fn test_fft() {
        const LOG_DEGREE: usize = 10;
        const DEGREE: usize = 1 << LOG_DEGREE;
        let p = DensePolynomial::from_iter((0..DEGREE).map(|_| PrimeField64::random(thread_rng())));
        let generator = PrimeField64::generator_for_fft(LOG_DEGREE);
        let output = fft(p.coefficients.as_ref());
        let orig = ifft(&output);
        let expected_outputs: Vec<_> = PowersIterator::new(generator)
            .take(DEGREE)
            .map(|v| p.eval(v))
            .collect();
        assert_eq!(output, expected_outputs);
        assert_eq!(orig, p.coefficients);
    }
    #[test]
    fn test_poly_mul() {
        const TESTS: usize = 100;
        const LOG_DEGREE: usize = 10;
        const DEGREE: usize = 1 << LOG_DEGREE;
        let mut rng = thread_rng();
        let p = DensePolynomial::from_iter((0..DEGREE).map(|_| PrimeField64::random(thread_rng())));
        let q = DensePolynomial::from_iter((0..DEGREE).map(|_| PrimeField64::random(thread_rng())));
        let p_clone = p.clone();
        let q_clone = q.clone();
        let s = &p_clone * &q_clone;
        for _ in 0..TESTS {
            let e = PrimeField64::random(&mut rng);
            assert_eq!(s.eval(e), p.eval(e) * q.eval(e));
        }
    }
}
