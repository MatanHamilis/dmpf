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
    coefficients: Vec<(usize, F)>,
}

impl<F: RadixTwoFftFriendFieldElement> SparsePolynomial<F> {
    pub fn new(coefficients: Vec<(usize, F)>) -> Self {
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
    pub fn random_regular<R: RngCore + CryptoRng>(
        mut rng: R,
        degree_bound: usize,
        weight: usize,
    ) -> Self {
        assert!(degree_bound > weight + 1);
        let slice_size = degree_bound / weight;
        let coefficients = (0..weight)
            .map(|i| {
                let coeff_deg = ((i * slice_size) + ((rng.next_u32() as usize) % slice_size))
                    .min(degree_bound - 1);
                let coeff = F::random(&mut rng);
                (coeff_deg, coeff)
            })
            .collect();
        SparsePolynomial::new(coefficients)
    }
    pub fn to_dense(&self) -> DensePolynomial<F> {
        let deg = self.coefficients.iter().map(|v| v.0).max().unwrap();
        let mut dense_coefficients = vec![F::zero(); deg + 1];
        self.coefficients
            .iter()
            .for_each(|(idx, f)| dense_coefficients[*idx] = *f);
        DensePolynomial {
            coefficients: dense_coefficients,
        }
    }
    fn is_regular(&self, degree_max: usize) -> bool {
        let weight = self.coefficients.len();
        let block_size = degree_max / weight;
        self.coefficients
            .iter()
            .enumerate()
            .all(|(i, (c, _))| (*c >= i * block_size) && (*c < i * block_size))
    }
}
// This function takes a polynomial of degree N with at most t*t non-zero points.
// In particular, this polynomial is a product of two regular, weight t, degree N polynomials.
// As such, the coefficients of this product polynomial are split into 2*t-1 sets where the i-th set
// considers contains min(i+1, 2*t-1-i) non-zero points of coefficients x^(N/(2t))
pub fn share_regular_polynomial_mul<
    const W: usize,
    F: RadixTwoFftFriendFieldElement,
    C: SmallFieldContainer<W, F>,
    D: Dmpf<C>,
>(
    p: &SparsePolynomial<F>,
    dmpf: &D,
    sqrt_points: usize,
    log_degree: usize,
) -> ((usize, Vec<(usize, D::Key)>), (usize, Vec<(usize, D::Key)>)) {
    let total_degree: usize = 1 << log_degree;
    let t = sqrt_points;
    let block_size = (total_degree / 2) / t;
    let block_dpf_domain_size = block_size.next_power_of_two();
    let log_sub_poly_degree = block_dpf_domain_size.ilog2() as usize;
    let mut total_blocks = 2 * t - 1;
    if total_blocks * block_size < total_degree {
        total_blocks += 1;
    }
    let mut coefficients_idx = 0;
    let mut new_poly_to_share = SparsePolynomial::<F>::new(Vec::with_capacity(2 * total_blocks));
    let mut coefficients_map = HashMap::with_capacity(2 * total_blocks);
    let (v_0, v_1): (Vec<_>, Vec<_>) = (0..total_blocks)
        .map(|block_idx| {
            let block_min_deg = block_idx * block_size;
            let block_max_deg = (block_idx + 1) * block_size;
            let expected_block_size = 2 * (block_idx).min(total_blocks - block_idx) + 1;
            while coefficients_idx < p.coefficients.len()
                && p.coefficients[coefficients_idx].0 < block_max_deg
            {
                let (idx, f) = p.coefficients[coefficients_idx];
                coefficients_map.insert(idx - block_min_deg, f);
                coefficients_idx += 1;
            }
            assert!(coefficients_map.len() <= expected_block_size);
            let mut i = block_min_deg;
            while coefficients_map.len() < expected_block_size && i < block_max_deg {
                if !coefficients_map.contains_key(&(i - block_min_deg)) {
                    coefficients_map.insert(i - block_min_deg, F::zero());
                }
                i += 1;
            }
            coefficients_map
                .iter()
                .for_each(|(k, v)| new_poly_to_share.coefficients.push((*k, *v)));
            new_poly_to_share.coefficients.sort_by_cached_key(|v| v.0);
            let (k_0, k_1) = share_polynomial(
                &new_poly_to_share,
                expected_block_size,
                dmpf,
                log_sub_poly_degree,
            );
            new_poly_to_share.coefficients.clear();
            coefficients_map.clear();
            ((block_min_deg, k_0), (block_min_deg, k_1))
        })
        .unzip();
    ((block_size, v_0), (block_size, v_1))
}
pub fn from_share_regular_error<
    const W: usize,
    F: RadixTwoFftFriendFieldElement,
    C: SmallFieldContainer<W, F>,
    D: Dmpf<C>,
>(
    key: (usize, &[(usize, D::Key)]),
    output_degree: usize,
) -> DensePolynomial<F> {
    let mut output = Vec::with_capacity(output_degree);
    unsafe { output.set_len(output_degree) };
    let (block_size, blocks) = key;
    blocks.iter().for_each(|(start_idx, key)| {
        let p = from_share::<W, F, C, D>(key);
        for i in 0..block_size.min(output_degree - *start_idx) {
            output[start_idx + i] = p[i]
        }
    });
    DensePolynomial {
        coefficients: output,
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
    for (idx, v) in p.coefficients.iter() {
        let cur_idx = ((idx >> lower_bits_to_shrink) as u128) << (128 - dpf_input_length);
        let internal_idx = idx & ((1 << lower_bits_to_shrink) - 1);
        let container = inputs_hashmap.entry(cur_idx).or_default();
        container[internal_idx] = *v;
    }
    let mut rng = thread_rng();
    let points = points.min(1 << dpf_input_length);
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

#[derive(Debug, Clone)]
pub struct DensePolynomial<F: RadixTwoFftFriendFieldElement> {
    pub coefficients: Vec<F>,
}
impl<F: RadixTwoFftFriendFieldElement> PartialEq for DensePolynomial<F> {
    fn eq(&self, other: &Self) -> bool {
        let min_len = self.coefficients.len().min(other.coefficients.len());
        let first_check = self.coefficients[..min_len] == other.coefficients[..min_len];
        let second_check = self.coefficients[min_len..].iter().all(|v| v.is_zero());
        let third_check = other.coefficients[min_len..].iter().all(|v| v.is_zero());
        first_check && second_check && third_check
    }
}
impl<F: RadixTwoFftFriendFieldElement> Eq for DensePolynomial<F> {}

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
            for (idx_sparse, c_sparse) in rhs.coefficients.iter() {
                output[idx + idx_sparse] += c * *c_sparse;
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
        let mut hash_output =
            HashMap::<usize, F>::with_capacity(self.coefficients.len() * rhs.coefficients.len());
        for (idx, c) in self.coefficients.iter() {
            for (idx_sparse, c_sparse) in rhs.coefficients.iter() {
                let v = hash_output.entry(idx + idx_sparse).or_insert(F::zero());
                *v += *c * *c_sparse;
            }
        }
        let mut coefficients: Vec<_> = hash_output.into_iter().collect();
        coefficients.sort_by_cached_key(|f| f.0);
        SparsePolynomial {
            coefficients: coefficients,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{
        fft::{fft, ifft},
        polynomial::{
            from_share, from_share_regular_error, share_polynomial, Polynomial, SparsePolynomial,
        },
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

    use super::{share_regular_polynomial_mul, DensePolynomial};

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
        let dmpf = OkvsDmpf::<100, PrimeField64x2>::new(dmpf::EpsilonPercent::Ten);
        // for _ in 0..TESTS {
        let mut seed = [0u8; 16];
        rng.fill_bytes(&mut seed);
        // let p = SparsePolynomial::random(&mut rng, DEGREE, WEIGHT);
        let p = SparsePolynomial::new(Vec::from_iter(
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
        p.coefficients.iter().for_each(|(idx, f)| {
            assert_eq!(coeffs[*idx], *f);
        })
        // }
    }
    #[test]
    fn test_regular_share() {
        let mut rng = thread_rng();
        const LOG_DEGREE_BOUND: usize = 9;
        const DEGREE_BOUND: usize = 1 << LOG_DEGREE_BOUND;
        const T: usize = 17;
        let dmpf = DpfDmpf::new();
        let p = SparsePolynomial::<PrimeField64>::random_regular(&mut rng, DEGREE_BOUND, T);
        let q = SparsePolynomial::<PrimeField64>::random_regular(&mut rng, DEGREE_BOUND, T);
        let pq = &p * &q;
        let (k_0, k_1) = share_regular_polynomial_mul::<2, _, PrimeField64x2, _>(
            &pq,
            &dmpf,
            T,
            LOG_DEGREE_BOUND + 1,
        );
        let reconstructed_first =
            from_share_regular_error::<2, PrimeField64, PrimeField64x2, DpfDmpf>(
                (k_0.0, &k_0.1),
                DEGREE_BOUND << 1,
            );
        let reconstructed_second =
            from_share_regular_error::<2, PrimeField64, PrimeField64x2, DpfDmpf>(
                (k_1.0, &k_1.1),
                DEGREE_BOUND << 1,
            );
        let sum = &reconstructed_first + &reconstructed_second;
        let pq_dense = pq.to_dense();
        assert_eq!(sum, pq_dense);
    }
}
