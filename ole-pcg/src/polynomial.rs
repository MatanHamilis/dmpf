use std::{
    collections::{HashMap, HashSet},
    ops::{Add, AddAssign, Index, IndexMut, Mul, Sub, SubAssign},
};

use crate::fft::{fft, ifft};
use dmpf::{
    field::{PowersIterator, RadixTwoFftFriendFieldElement},
    Dmpf,
};
use dmpf::{DmpfKey, SmallFieldContainer};
use rand::{thread_rng, CryptoRng, RngCore};
pub trait Polynomial<F: RadixTwoFftFriendFieldElement>:
    Add + AddAssign + SubAssign + Clone + Index<usize> + IndexMut<usize>
where
    for<'a> &'a Self: Mul<Output = Self>,
    for<'a> &'a Self: Sub<Output = Self>,
{
    fn eval(&self, input: F) -> F;
}

/// This implements a sparse polynomial represented as coefficients
#[derive(Debug)]
pub struct SparsePolynomial<F: RadixTwoFftFriendFieldElement> {
    coefficients: HashMap<usize, F>,
}

impl<F: RadixTwoFftFriendFieldElement> SparsePolynomial<F> {
    pub fn new(coefficients: HashMap<usize, F>) -> Self {
        Self { coefficients }
    }
    pub fn random<R: RngCore + CryptoRng>(mut rng: R, degree: usize, weight: usize) -> Self {
        assert!(degree > weight + 1);
        let mut hash_set = HashSet::new();
        while hash_set.len() < weight {
            hash_set.insert((rng.next_u64() % (degree as u64)) as usize);
        }
        let coefficients = hash_set.into_iter().map(|idx| (idx, F::random(&mut rng)));
        SparsePolynomial::new(coefficients.collect())
    }
    pub fn to_dense(&self) -> DensePolynomial<F> {
        let deg = self.coefficients.iter().map(|v| *v.0).max().unwrap();
        let mut dense_coefficients = vec![F::zero(); deg + 1];
        self.coefficients
            .iter()
            .for_each(|(idx, f)| dense_coefficients[*idx] = *f);
        DensePolynomial {
            coefficients: dense_coefficients,
        }
    }
}
pub fn share_polynomial<
    const W: usize,
    F: RadixTwoFftFriendFieldElement,
    C: SmallFieldContainer<W, F>,
    D: Dmpf<C>,
>(
    p: &SparsePolynomial<F>,
    points: usize,
    dmpf: &D,
    log_degree: usize,
) -> (D::Key, D::Key) {
    assert!(points >= p.coefficients.len());
    assert!(W.is_power_of_two());
    let lower_bits_to_shrink = W.ilog2() as usize;
    let dpf_input_length = log_degree - lower_bits_to_shrink;
    let mut inputs_hashmap = HashMap::<u128, C>::new();
    for (&idx, &v) in p.coefficients.iter() {
        let cur_idx = ((idx >> lower_bits_to_shrink) as u128) << (128 - dpf_input_length);
        let internal_idx = idx & ((1 << lower_bits_to_shrink) - 1);
        let container = inputs_hashmap.entry(cur_idx).or_default();
        container[internal_idx] = v;
    }
    let mut rng = thread_rng();
    while inputs_hashmap.len() < points {
        let idx = rng.next_u64() % (1 << log_degree);
        let cur_idx = ((idx >> lower_bits_to_shrink) as u128) << (128 - dpf_input_length);
        if inputs_hashmap.contains_key(&cur_idx) {
            continue;
        }
        inputs_hashmap.entry(cur_idx).or_default();
    }
    let mut v: Vec<_> = inputs_hashmap.into_iter().collect();
    v.sort_unstable_by_key(|f| f.0);
    dmpf.try_gen(log_degree - lower_bits_to_shrink, &v[..], &mut thread_rng())
        .unwrap()
}

pub fn from_share<
    const W: usize,
    F: RadixTwoFftFriendFieldElement,
    C: SmallFieldContainer<W, F>,
    D: Dmpf<C>,
>(
    key: &D::Key,
) -> DensePolynomial<F> {
    let evals = key.eval_all();
    let mut coefficients = Vec::<F>::with_capacity(evals.len() * W);
    evals.iter().for_each(|c| {
        for i in 0..W {
            coefficients.push(c[i]);
        }
    });
    DensePolynomial { coefficients }
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    pub fn random<R: RngCore + CryptoRng>(mut rng: R, degree: usize) -> Self {
        Self::from_iter((0..degree).map(|_| F::random(&mut rng)))
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
    fn sub_assign(&mut self, rhs: Self) {
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

impl<F: RadixTwoFftFriendFieldElement> AddAssign<&DensePolynomial<F>> for DensePolynomial<F> {
    fn add_assign(&mut self, rhs: &Self) {
        debug_assert_eq!(self.coefficients.len(), rhs.coefficients.len());
        self.coefficients
            .iter_mut()
            .zip(rhs.coefficients.iter())
            .for_each(|(o, i)| *o += *i);
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
impl<F: RadixTwoFftFriendFieldElement> Add for &DensePolynomial<F> {
    type Output = DensePolynomial<F>;
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.coefficients.len(), rhs.coefficients.len());
        DensePolynomial::from_iter(
            self.coefficients
                .iter()
                .zip(rhs.coefficients.iter())
                .map(|(a, b)| *a + *b),
        )
    }
}

impl<F: RadixTwoFftFriendFieldElement> Mul for &DensePolynomial<F> {
    type Output = DensePolynomial<F>;
    fn mul(self, rhs: &DensePolynomial<F>) -> Self::Output {
        assert_eq!(self.coefficients.len(), rhs.coefficients.len());
        let log_dest_size = ((self.coefficients.len() * 2 - 1).ilog2() + 1) as usize;
        let dest_size: usize = 1 << log_dest_size;
        let mut self_c = self.coefficients.clone();
        self_c.resize(dest_size, F::zero());
        let mut rhs_c = rhs.coefficients.clone();
        rhs_c.resize(dest_size, F::zero());
        let mut self_evals = fft(&self_c);
        let rhs_evals = fft(&rhs_c);
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
        for (idx, c) in self.coefficients.iter().copied().enumerate() {
            for (&idx_sparse, &c_sparse) in rhs.coefficients.iter() {
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
        let mut hash_output = HashMap::<usize, F>::new();
        for (&idx, &c) in self.coefficients.iter() {
            for (&idx_sparse, &c_sparse) in rhs.coefficients.iter() {
                let v = hash_output.entry(idx + idx_sparse).or_insert(F::zero());
                *v += c * c_sparse;
            }
        }
        SparsePolynomial {
            coefficients: hash_output,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{
        fft::{fft, ifft},
        polynomial::{from_share, share_polynomial, Polynomial, SparsePolynomial},
        ring::TwoPowerDegreeCyclotomicPolynomial,
    };
    use dmpf::{
        field::{
            FieldElement, PowersIterator, PrimeField64, PrimeField64x2,
            RadixTwoFftFriendFieldElement,
        },
        okvs::OkvsDmpf,
        DpfDmpf,
    };
    use rand::{thread_rng, RngCore};

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
        let s = &p * &q;
        for _ in 0..TESTS {
            let e = PrimeField64::random(&mut rng);
            assert_eq!(s.eval(e), p.eval(e) * q.eval(e));
        }
    }
    #[test]
    fn test_share() {
        const TESTS: usize = 100;
        const LOG_DEGREE: usize = 3;
        const DEGREE: usize = 1 << LOG_DEGREE;
        const WEIGHT: usize = 2;
        let mut rng = thread_rng();
        let dmpf = OkvsDmpf::<100, PrimeField64x2>::new(WEIGHT, dmpf::EpsilonPercent::Ten);
        // for _ in 0..TESTS {
        let mut seed = [0u8; 16];
        rng.fill_bytes(&mut seed);
        // let p = SparsePolynomial::random(&mut rng, DEGREE, WEIGHT);
        let p = SparsePolynomial::new(HashMap::from_iter(
            vec![
                (0, PrimeField64::random(&mut rng)),
                (1, (PrimeField64::random(&mut rng))),
            ]
            .into_iter(),
        ));
        let (share_a, share_b) = share_polynomial(&p, WEIGHT, &dmpf, LOG_DEGREE);
        let (eval_a, eval_b) = (
            from_share::<2, _, PrimeField64x2, OkvsDmpf<100, PrimeField64x2>>(&share_a),
            from_share::<2, _, PrimeField64x2, OkvsDmpf<100, PrimeField64x2>>(&share_b),
        );
        let mut reconstructed = eval_a + eval_b;
        let coeffs = &mut reconstructed.coefficients;
        p.coefficients.iter().for_each(|(&idx, &f)| {
            assert_eq!(coeffs[idx], f);
        })
        // }
    }
}
