use std::{
    collections::HashSet,
    ops::{Index, IndexMut, Mul},
};

use aes_prng::AesRng;
use dmpf::RadixTwoFftFriendFieldElement;
use polynomial::{DensePolynomial, SparsePolynomial};
use rand::{thread_rng, RngCore, SeedableRng};
use ring::{ModuloPolynomial, PolynomialRingElement, SparsePolynomialRingElement};

mod fft;
mod field;
mod polynomial;
mod ring;
enum Role {
    First,
    Second,
}
pub struct OlePcgSeed<F: RadixTwoFftFriendFieldElement, D: Dmpf> {
    my_vec_seed: [u8; 16],
    peer_vec_seed: [u8; 16],
    role: Role,
    compression_factor: usize,
    sparse_polynomial: SparsePolynomial<F>,
    tensor_product_fss: D::Key,
}

fn polynomials_from_seed<F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>>(
    seed: [u8; 16],
    log_degree: usize,
    modulo_polynomial: M,
    polyomials_count: usize,
) -> Vec<PolynomialRingElement<F, M>> {
    let mut rng = AesRng::from_seed(seed);
    let degree = 1 << log_degree;
    (0..polyomials_count)
        .map(|_| {
            let p = DensePolynomial::<F>::from_iter((0..degree).map(|_| F::random(&mut rng)));
            PolynomialRingElement::new(p, modulo_polynomial)
        })
        .collect()
}

fn sparse_polynomials_from_seed<F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>>(
    seed: [u8; 16],
    log_degree: usize,
    weight: usize,
    modulo_polynomial: M,
    polyomials_count: usize,
) -> Vec<SparsePolynomialRingElement<F, M>> {
    let mut rng = AesRng::from_seed(seed);
    let degree = 1 << log_degree;
    (0..polyomials_count)
        .map(|_| {
            let mut hash_set = HashSet::new();
            while hash_set.len() < weight {
                hash_set.insert((rng.next_u64() % degree) as usize);
            }
            let coefficients = hash_set
                .into_iter()
                .map(|idx| (F::random(&mut rng), idx))
                .collect();
            SparsePolynomialRingElement::new(SparsePolynomial::new(coefficients), modulo_polynomial)
        })
        .collect()
}
pub struct Matrix<T> {
    v: Vec<T>,
    row_len: usize,
}
impl<T> Matrix<T> {
    pub fn new(v: Vec<T>, row_len: usize) -> Self {
        assert_eq!(v.len() % row_len, 0);
        Self { v, row_len }
    }
}
impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.v[index.0 * self.row_len + index.1]
    }
}
impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.v[index.0 * self.row_len + index.1]
    }
}

fn tensor_product<'a, 'b, T>(a: &'a [T], b: &'b [T]) -> Matrix<<&'a T as Mul<&'b T>>::Output>
where
    &'a T: Mul<&'b T>,
{
    let mut output = Vec::<<&'a T as Mul<&'b T>>::Output>::with_capacity(a.len() * b.len());
    for ai in a {
        for bi in b {
            output.push(ai * bi);
        }
    }
    Matrix::new(output, b.len())
}

fn gen<F: RadixTwoFftFriendFieldElement, D: Dmpf, M: ModuloPolynomial<F>>(
    log_polynomial_degree: usize,
    compression_factor: usize,
    modulo_polynomial: M,
    weight: usize,
) -> (OlePcgSeed<F, D>, OlePcgSeed<F, D>) {
    let mut rng = thread_rng();
    let mut first_polynomial_seed = [0u8; 16];
    let mut second_polynomial_seed = [0u8; 16];
    let mut first_sparse_seed = [0u8; 16];
    let mut second_sparse_seed = [0u8; 16];
    rng.fill_bytes(&mut first_polynomial_seed);
    rng.fill_bytes(&mut second_polynomial_seed);
    rng.fill_bytes(&mut first_sparse_seed);
    rng.fill_bytes(&mut second_sparse_seed);
    let first_polynomials = polynomials_from_seed(
        first_polynomial_seed,
        log_polynomial_degree,
        modulo_polynomial.clone(),
        compression_factor,
    );
    let second_polynomials = polynomials_from_seed(
        second_polynomial_seed,
        log_polynomial_degree,
        modulo_polynomial.clone(),
        compression_factor,
    );
    let first_sparse_polynomials = sparse_polynomials_from_seed(
        first_sparse_seed,
        log_polynomial_degree,
        weight,
        modulo_polynomial,
        compression_factor,
    );
    let second_sparse_polynomials = sparse_polynomials_from_seed(
        second_sparse_seed,
        log_polynomial_degree,
        weight,
        modulo_polynomial,
        compression_factor,
    );
    let tensor_product = tensor_product(&first_sparse_polynomials, &second_sparse_polynomials);
    let mut first_tensor_shares = Vec::with_capacity(compression_factor * compression_factor);
    let mut second_tensor_shares = Vec::with_capacity(compression_factor * compression_factor);
    tensor_product.v.into_iter().map(|p| {})
}
