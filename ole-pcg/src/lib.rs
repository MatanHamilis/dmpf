use std::{
    ops::{Index, IndexMut, Mul},
    time::Instant,
};

use aes_prng::AesRng;
use dmpf::{Dmpf, RadixTwoFftFriendFieldElement, SmallFieldContainer};
use polynomial::{
    share_polynomial, share_regular_polynomial_mul, DensePolynomial, SparsePolynomial,
};
use rand::{thread_rng, RngCore, SeedableRng};
use ring::{ModuloPolynomial, PolynomialRingElement, SparsePolynomialRingElement};

use crate::polynomial::{from_share, from_share_regular_error};

mod fft;
mod polynomial;
pub mod ring;
enum Role {
    First,
    Second,
}
pub struct OlePcgSeed<
    const CONTAINER_SIZE: usize,
    F: RadixTwoFftFriendFieldElement,
    C: SmallFieldContainer<CONTAINER_SIZE, F>,
    D: Dmpf<C>,
    M: ModuloPolynomial<F>,
> {
    my_vec_seed: [u8; 16],
    peer_vec_seed: [u8; 16],
    role: Role,
    compression_factor: usize,
    sparse_polynomials: Vec<SparsePolynomialRingElement<F, M>>,
    tensor_product_fss: Vec<D::Key>,
}
impl<
        const CONTAINER_SIZE: usize,
        F: RadixTwoFftFriendFieldElement,
        C: SmallFieldContainer<CONTAINER_SIZE, F>,
        D: Dmpf<C>,
        M: ModuloPolynomial<F>,
    > OlePcgSeed<CONTAINER_SIZE, F, C, D, M>
{
    fn new(
        my_vec_seed: [u8; 16],
        peer_vec_seed: [u8; 16],
        role: Role,
        compression_factor: usize,
        sparse_polynomials: Vec<SparsePolynomialRingElement<F, M>>,
        tensor_product_fss: Vec<D::Key>,
    ) -> Self {
        Self {
            my_vec_seed,
            peer_vec_seed,
            role,
            compression_factor,
            sparse_polynomials,
            tensor_product_fss,
        }
    }
    pub fn expand(&self) -> (PolynomialRingElement<F, M>, PolynomialRingElement<F, M>) {
        // First, compute the multiplicative shares:
        let my_polynomials = polynomials_from_seed(
            self.my_vec_seed,
            self.sparse_polynomials[0].modulo().clone(),
            self.compression_factor,
        );
        let peer_polynomials = polynomials_from_seed(
            self.peer_vec_seed,
            self.sparse_polynomials[0].modulo().clone(),
            self.compression_factor,
        );
        let time = Instant::now();
        assert_eq!(my_polynomials.len(), self.sparse_polynomials.len());
        let multiplicative_share_poly: PolynomialRingElement<F, M> = my_polynomials
            .iter()
            .zip(self.sparse_polynomials.iter())
            .map(|(p, s)| s * p)
            .sum();
        println!("Multiplicative share gen: {}", time.elapsed().as_millis());

        // Second, compute additive shares:
        let tensor_product_dense = match self.role {
            Role::First => tensor_product(&my_polynomials, &peer_polynomials),
            Role::Second => tensor_product(&peer_polynomials, &my_polynomials),
        };
        let time = Instant::now();
        let additive_share_poly: PolynomialRingElement<F, M> = self
            .tensor_product_fss
            .iter()
            .zip(tensor_product_dense.v.iter())
            .map(|(k, p)| {
                let dense_poly = from_share::<CONTAINER_SIZE, F, C, D>(k);
                let dense_poly = p.get_modulo().modulo(dense_poly);
                let dense_ring_element =
                    PolynomialRingElement::new(dense_poly, p.get_modulo().clone());
                p * &dense_ring_element
            })
            .sum();
        println!("Additive share gen: {}", time.elapsed().as_millis());
        (multiplicative_share_poly, additive_share_poly)
    }
}

pub struct OlePcgSeedRegular<
    const CONTAINER_SIZE: usize,
    F: RadixTwoFftFriendFieldElement,
    C: SmallFieldContainer<CONTAINER_SIZE, F>,
    D: Dmpf<C>,
    M: ModuloPolynomial<F>,
> {
    my_vec_seed: [u8; 16],
    peer_vec_seed: [u8; 16],
    role: Role,
    compression_factor: usize,
    sparse_polynomials: Vec<SparsePolynomialRingElement<F, M>>,
    tensor_product_fss: Vec<(usize, Vec<(usize, D::Key)>)>,
}
impl<
        const CONTAINER_SIZE: usize,
        F: RadixTwoFftFriendFieldElement,
        C: SmallFieldContainer<CONTAINER_SIZE, F>,
        D: Dmpf<C>,
        M: ModuloPolynomial<F>,
    > OlePcgSeedRegular<CONTAINER_SIZE, F, C, D, M>
{
    fn new(
        my_vec_seed: [u8; 16],
        peer_vec_seed: [u8; 16],
        role: Role,
        compression_factor: usize,
        sparse_polynomials: Vec<SparsePolynomialRingElement<F, M>>,
        tensor_product_fss: Vec<(usize, Vec<(usize, D::Key)>)>,
    ) -> Self {
        Self {
            my_vec_seed,
            peer_vec_seed,
            role,
            compression_factor,
            sparse_polynomials,
            tensor_product_fss,
        }
    }
    pub fn expand(&self) -> (PolynomialRingElement<F, M>, PolynomialRingElement<F, M>) {
        // First, compute the multiplicative shares:
        let my_polynomials = polynomials_from_seed(
            self.my_vec_seed,
            self.sparse_polynomials[0].modulo().clone(),
            self.compression_factor,
        );
        let peer_polynomials = polynomials_from_seed(
            self.peer_vec_seed,
            self.sparse_polynomials[0].modulo().clone(),
            self.compression_factor,
        );
        let time = Instant::now();
        assert_eq!(my_polynomials.len(), self.sparse_polynomials.len());
        let multiplicative_share_poly: PolynomialRingElement<F, M> = my_polynomials
            .iter()
            .zip(self.sparse_polynomials.iter())
            .map(|(p, s)| s * p)
            .sum();
        println!("Multiplicative share gen: {}", time.elapsed().as_millis());

        // Second, compute additive shares:
        let tensor_product_dense = match self.role {
            Role::First => tensor_product(&my_polynomials, &peer_polynomials),
            Role::Second => tensor_product(&peer_polynomials, &my_polynomials),
        };
        let time = Instant::now();
        assert_eq!(
            self.tensor_product_fss.len(),
            self.compression_factor * self.compression_factor
        );
        let additive_share_poly: PolynomialRingElement<F, M> = self
            .tensor_product_fss
            .iter()
            .zip(tensor_product_dense.v.iter())
            .map(|((k_0, k_1), p)| {
                let dense_poly = from_share_regular_error::<CONTAINER_SIZE, F, C, D>(
                    (*k_0, &k_1[..]),
                    self.sparse_polynomials[0].modulo().deg() * 2,
                );
                let dense_poly = p.get_modulo().modulo(dense_poly);
                let dense_ring_element =
                    PolynomialRingElement::new(dense_poly, p.get_modulo().clone());
                p * &dense_ring_element
            })
            .sum();
        println!("Additive share gen: {}", time.elapsed().as_millis());
        (multiplicative_share_poly, additive_share_poly)
    }
}

fn polynomials_from_seed<F: RadixTwoFftFriendFieldElement, M: ModuloPolynomial<F>>(
    seed: [u8; 16],
    modulo_polynomial: M,
    polyomials_count: usize,
) -> Vec<PolynomialRingElement<F, M>> {
    let mut rng = AesRng::from_seed(seed);
    let degree = modulo_polynomial.deg();
    (0..polyomials_count)
        .map(|_| {
            let p = DensePolynomial::<F>::from_iter((0..degree).map(|_| F::random(&mut rng)));
            PolynomialRingElement::new(p, modulo_polynomial.clone())
        })
        .collect()
}

fn sparse_regular_polynomials_from_seed<F: RadixTwoFftFriendFieldElement>(
    seed: [u8; 16],
    weight: usize,
    degree: usize,
    polyomials_count: usize,
) -> Vec<SparsePolynomial<F>> {
    let mut rng = AesRng::from_seed(seed);
    (0..polyomials_count)
        .map(|_| SparsePolynomial::random_regular(&mut rng, degree, weight))
        .collect()
}

fn sparse_polynomials_from_seed<F: RadixTwoFftFriendFieldElement>(
    seed: [u8; 16],
    weight: usize,
    degree: usize,
    polyomials_count: usize,
) -> Vec<SparsePolynomial<F>> {
    let mut rng = AesRng::from_seed(seed);
    (0..polyomials_count)
        .map(|_| SparsePolynomial::random(&mut rng, degree, weight))
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

pub fn gen_regular<
    const CONTAINER_SIZE: usize,
    F: RadixTwoFftFriendFieldElement,
    C: SmallFieldContainer<CONTAINER_SIZE, F>,
    D: Dmpf<C>,
    M: ModuloPolynomial<F>,
>(
    log_polynomial_degree: usize,
    compression_factor: usize,
    modulo_polynomial: M,
    weight: usize,
    dmpf: &D,
) -> (
    OlePcgSeedRegular<CONTAINER_SIZE, F, C, D, M>,
    OlePcgSeedRegular<CONTAINER_SIZE, F, C, D, M>,
) {
    let mut rng = thread_rng();
    let mut first_polynomial_seed = [0u8; 16];
    // let mut second_polynomial_seed = [0u8; 16];
    let mut first_sparse_seed = [0u8; 16];
    let mut second_sparse_seed = [0u8; 16];
    rng.fill_bytes(&mut first_polynomial_seed);
    let second_polynomial_seed = first_polynomial_seed;
    // rng.fill_bytes(&mut second_polynomial_seed);
    rng.fill_bytes(&mut first_sparse_seed);
    rng.fill_bytes(&mut second_sparse_seed);
    let first_sparse_polynomials = sparse_regular_polynomials_from_seed(
        first_sparse_seed,
        weight,
        modulo_polynomial.deg(),
        compression_factor,
    );
    let second_sparse_polynomials = sparse_regular_polynomials_from_seed::<F>(
        second_sparse_seed,
        weight,
        modulo_polynomial.deg(),
        compression_factor,
    );
    let tensor_product = tensor_product(&first_sparse_polynomials, &second_sparse_polynomials);
    let first_sparse_polynomials: Vec<_> = first_sparse_polynomials
        .into_iter()
        .map(|p| SparsePolynomialRingElement::new(p, modulo_polynomial.clone()))
        .collect();
    let second_sparse_polynomials: Vec<_> = second_sparse_polynomials
        .into_iter()
        .map(|p| SparsePolynomialRingElement::new(p, modulo_polynomial.clone()))
        .collect();
    let (first_shares, second_shares): (Vec<_>, Vec<_>) = tensor_product
        .v
        .into_iter()
        .map(|p| share_regular_polynomial_mul(&p, dmpf, weight, log_polynomial_degree + 1))
        .unzip();
    let first = OlePcgSeedRegular::new(
        first_polynomial_seed,
        second_polynomial_seed,
        Role::First,
        compression_factor,
        first_sparse_polynomials,
        first_shares,
    );
    let second = OlePcgSeedRegular::new(
        second_polynomial_seed,
        first_polynomial_seed,
        Role::Second,
        compression_factor,
        second_sparse_polynomials,
        second_shares,
    );
    (first, second)
}
pub fn gen<
    const CONTAINER_SIZE: usize,
    F: RadixTwoFftFriendFieldElement,
    C: SmallFieldContainer<CONTAINER_SIZE, F>,
    D: Dmpf<C>,
    M: ModuloPolynomial<F>,
>(
    log_polynomial_degree: usize,
    compression_factor: usize,
    modulo_polynomial: M,
    weight: usize,
    dmpf: &D,
) -> (
    OlePcgSeed<CONTAINER_SIZE, F, C, D, M>,
    OlePcgSeed<CONTAINER_SIZE, F, C, D, M>,
) {
    let mut rng = thread_rng();
    let mut first_polynomial_seed = [0u8; 16];
    // let mut second_polynomial_seed = [0u8; 16];
    let mut first_sparse_seed = [0u8; 16];
    let mut second_sparse_seed = [0u8; 16];
    rng.fill_bytes(&mut first_polynomial_seed);
    let second_polynomial_seed = first_polynomial_seed;
    // rng.fill_bytes(&mut second_polynomial_seed);
    rng.fill_bytes(&mut first_sparse_seed);
    rng.fill_bytes(&mut second_sparse_seed);
    let first_sparse_polynomials = sparse_polynomials_from_seed(
        first_sparse_seed,
        weight,
        modulo_polynomial.deg(),
        compression_factor,
    );
    let second_sparse_polynomials = sparse_polynomials_from_seed::<F>(
        second_sparse_seed,
        weight,
        modulo_polynomial.deg(),
        compression_factor,
    );
    let tensor_product = tensor_product(&first_sparse_polynomials, &second_sparse_polynomials);
    let first_sparse_polynomials: Vec<_> = first_sparse_polynomials
        .into_iter()
        .map(|p| SparsePolynomialRingElement::new(p, modulo_polynomial.clone()))
        .collect();
    let second_sparse_polynomials: Vec<_> = second_sparse_polynomials
        .into_iter()
        .map(|p| SparsePolynomialRingElement::new(p, modulo_polynomial.clone()))
        .collect();
    let (first_shares, second_shares): (Vec<_>, Vec<_>) = tensor_product
        .v
        .into_iter()
        .map(|p| share_polynomial(&p, weight * weight, dmpf, log_polynomial_degree + 1))
        .unzip();
    let first = OlePcgSeed::new(
        first_polynomial_seed,
        second_polynomial_seed,
        Role::First,
        compression_factor,
        first_sparse_polynomials,
        first_shares,
    );
    let second = OlePcgSeed::new(
        second_polynomial_seed,
        first_polynomial_seed,
        Role::Second,
        compression_factor,
        second_sparse_polynomials,
        second_shares,
    );
    (first, second)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, time::Instant};

    use dmpf::{
        batch_code::BatchCodeDmpf, field::FieldElement, okvs::OkvsDmpf, DpfDmpf, EpsilonPercent,
        PrimeField64, PrimeField64x2,
    };
    use rand::thread_rng;

    use crate::{
        gen, gen_regular,
        polynomial::{from_share, from_share_regular_error, share_polynomial, SparsePolynomial},
        ring::{ModuloPolynomial, PolynomialRingElement, TwoPowerDegreeCyclotomicPolynomial},
    };

    #[test]
    fn test_multiplicative_sharing() {
        const LOG_DEGREE: usize = 10;
        const WEIGHT: usize = 20;
        let mut rng = thread_rng();
        let modulo = TwoPowerDegreeCyclotomicPolynomial::<PrimeField64>::new(LOG_DEGREE);

        let sparse_a = SparsePolynomial::<PrimeField64>::random(&mut rng, 1 << LOG_DEGREE, WEIGHT);
        let sparse_b = SparsePolynomial::random(&mut rng, 1 << LOG_DEGREE, WEIGHT);
        let sparse_ab = &sparse_a * &sparse_b;
        let dense_ab = sparse_ab.to_dense();
        let dense_ab_ring = PolynomialRingElement::new(modulo.modulo(dense_ab), modulo.clone());

        let dmpf = OkvsDmpf::new(EpsilonPercent::Ten);
        let (share_a, share_b) = share_polynomial::<2, _, PrimeField64x2, _>(
            &sparse_ab,
            WEIGHT * WEIGHT,
            &dmpf,
            LOG_DEGREE + 1,
        );
        let additive_a_poly = from_share::<2, _, PrimeField64x2, OkvsDmpf<400, _>>(&share_a);
        let additive_a = PolynomialRingElement::new(modulo.modulo(additive_a_poly), modulo.clone());
        let additive_b_poly = from_share::<2, _, PrimeField64x2, OkvsDmpf<400, _>>(&share_b);
        let additive_b = PolynomialRingElement::new(modulo.modulo(additive_b_poly), modulo.clone());
        let sum = &additive_a + &additive_b;
        assert_eq!(dense_ab_ring, sum);
    }
    #[test]
    fn test_ole_pcg() {
        const LOG_POLYNOMIAL_DEGREE: usize = 6;
        const COMPRESSION_FACTOR: usize = 2;
        const WEIGHT: usize = 5;
        // let dmpf = OkvsDmpf::<200, _>::new(EpsilonPercent::Ten);
        let dmpf = DpfDmpf::new();
        // let dmpf = BatchCodeDmpf::new(4, 50);
        let modulo_polynomial =
            TwoPowerDegreeCyclotomicPolynomial::<PrimeField64>::new(LOG_POLYNOMIAL_DEGREE);
        let time = Instant::now();
        let (first, second) = gen::<2, _, PrimeField64x2, _, _>(
            LOG_POLYNOMIAL_DEGREE,
            COMPRESSION_FACTOR,
            modulo_polynomial,
            WEIGHT,
            &dmpf,
        );
        println!("OLE PCG took: {}", time.elapsed().as_millis());
        let time = Instant::now();
        let first_shares = first.expand();
        println!("Expand took: {}", time.elapsed().as_millis());
        let time = Instant::now();
        let second_shares = second.expand();
        println!("Expand took: {}", time.elapsed().as_millis());

        assert_eq!(
            &first_shares.0 * &second_shares.0,
            &first_shares.1 + &second_shares.1
        );
    }
    #[test]
    fn test_ole_pcg_regular() {
        const LOG_POLYNOMIAL_DEGREE: usize = 10;
        const COMPRESSION_FACTOR: usize = 2;
        const WEIGHT: usize = 17;
        // let dmpf = OkvsDmpf::<200, _>::new(EpsilonPercent::Ten);
        let dmpf = DpfDmpf::new();
        // let dmpf = BatchCodeDmpf::new(4, 50);
        let modulo_polynomial =
            TwoPowerDegreeCyclotomicPolynomial::<PrimeField64>::new(LOG_POLYNOMIAL_DEGREE);
        let time = Instant::now();
        let (first, second) = gen_regular::<2, _, PrimeField64x2, _, _>(
            LOG_POLYNOMIAL_DEGREE,
            COMPRESSION_FACTOR,
            modulo_polynomial,
            WEIGHT,
            &dmpf,
        );
        println!("OLE PCG took: {}", time.elapsed().as_millis());

        let time = Instant::now();
        let first_shares = first.expand();
        println!("Expand took: {}", time.elapsed().as_millis());
        let time = Instant::now();
        let second_shares = second.expand();
        println!("Expand took: {}", time.elapsed().as_millis());

        let mul = &first_shares.0 * &second_shares.0;
        let add = &first_shares.1 + &second_shares.1;
        assert_eq!(mul, add);
    }
}
