use std::{
    fmt::Debug,
    hash::Hash,
    iter::Sum,
    marker::PhantomData,
    ops::{BitXor, BitXorAssign, Mul, SubAssign},
};

use aes_prng::AesRng;
use rand::{thread_rng, CryptoRng, RngCore, SeedableRng};

use crate::{random_u128, EpsilonPercent, LogN};

use super::{OkvsKey, OkvsU128};
impl BinaryOkvsValue for OkvsU128 {
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
    fn random<R: CryptoRng + RngCore>(mut rng: R) -> Self {
        Self(random_u128(&mut rng))
    }
}
impl Mul<bool> for OkvsU128 {
    type Output = Self;
    fn mul(mut self, rhs: bool) -> Self::Output {
        self.0 *= rhs as u128;
        self
    }
}
impl Default for OkvsU128 {
    fn default() -> Self {
        Self(0)
    }
}
impl BitXorAssign for OkvsU128 {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
impl BitXor for OkvsU128 {
    type Output = Self;
    fn bitxor(mut self, rhs: Self) -> Self::Output {
        self ^= rhs;
        self
    }
}
impl Sum for OkvsU128 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self(iter.fold(0u128, |a, b| a ^ b.0))
    }
}
impl From<bool> for OkvsU128 {
    fn from(value: bool) -> Self {
        Self(value as u128)
    }
}

pub trait BinaryOkvsValue:
    Default
    + Copy
    + Clone
    + Copy
    + Eq
    + PartialEq
    + Debug
    + Hash
    + BitXor<Output = Self>
    + BitXorAssign
    + From<bool>
    + Mul<bool, Output = Self>
    + Sum
{
    fn random<R: CryptoRng + RngCore>(rng: R) -> Self;
    fn is_zero(&self) -> bool;
}

#[derive(Debug, Clone, Copy)]
struct BinaryMatrixRow<const W: usize, V: BinaryOkvsValue> {
    first_col: usize,
    band: [u64; W],
    rhs: V,
}

impl<const W: usize, V: BinaryOkvsValue> BinaryMatrixRow<W, V> {
    fn from_key_value<K: OkvsKey>(key: &K, value: V, m: usize, base_seed: &[u8; 16]) -> Self {
        let (first_col, band) = hash_key(key, m, base_seed);
        let mut output = Self {
            band,
            first_col,
            rhs: value,
        };
        output.align();
        output
    }
    fn align(&mut self) {
        let shift = self.band[0].trailing_zeros();
        if shift == 64 {
            panic!();
        }
        debug_assert_ne!(shift, 64);
        let mask: u64 = u64::rotate_right((1 << shift) - 1, shift);
        let anti_mask: u64 = !mask;
        self.band[0] = self.band[0].overflowing_shr(shift).0;
        for i in 1..W {
            self.band[i] = self.band[i].rotate_right(shift);
            self.band[i - 1] ^= self.band[i] & mask;
            self.band[i] &= anti_mask;
        }
        self.first_col += shift as usize;
    }
    fn get_bit_value_uncheck(&self, column: usize) -> bool {
        let bit = column - self.first_col;
        let cell = bit / 64;
        let idx = bit & 63;
        (self.band[cell] >> idx) & 1 != 0
    }
}

// From RB-OKVS paper appendix F.
// Essentially for LAMBDA = 40, W ranges between 64-bit array of size 4 to 9.
pub const fn g(lambda: usize, epsilon_percent: EpsilonPercent, log_n: LogN) -> usize {
    super::g(lambda, epsilon_percent, log_n).div_ceil(64)
}

#[derive(Clone)]
pub struct BinaryEncodedOkvs<const W: usize, K: OkvsKey, V: BinaryOkvsValue>(
    Vec<V>,
    PhantomData<K>,
    [u8; 16],
);
impl<const W: usize, K: OkvsKey, V: BinaryOkvsValue> BinaryEncodedOkvs<W, K, V> {
    pub fn decode(&self, key: &K) -> V {
        let (offset, bits) = hash_key_compressed::<W, K>(key, self.0.len(), &self.2);
        let slice = &self.0[offset..offset + (W * 64 as usize)];
        bits.zip(slice)
            .map(|(bit, v)| *v * bit)
            .fold(V::default(), |cur, acc| cur ^ acc)
    }
}
impl<const W: usize, V: BinaryOkvsValue> SubAssign<&BinaryMatrixRow<W, V>>
    for BinaryMatrixRow<W, V>
{
    // Invariant: We always subtract only aligned rows (i.e. for which the band starts at the same column).
    fn sub_assign(&mut self, rhs: &Self) {
        // If I'm being subtracted, then anyway nothing should change *before* my first column.
        debug_assert_eq!(self.first_col, rhs.first_col);
        self.band
            .iter_mut()
            .zip(rhs.band.iter())
            .for_each(|(a, b)| *a ^= *b);
        self.rhs ^= rhs.rhs;
        self.align();
    }
}

/// We split the functions to avoid generating a large array when `decode`-ing.
pub fn hash_key_compressed<const W: usize, K: OkvsKey>(
    k: &K,
    m: usize,
    base_seed: &[u8; 16],
) -> (usize, impl Iterator<Item = bool>) {
    let block = k.hash_seed(base_seed);
    let mut seed = AesRng::from_seed(block);
    let mut cur_bits = 0;
    let band_start = (seed.next_u64() as usize) % (m - 64 * W);
    (
        band_start,
        (0..64 * W).map(move |i| {
            if (i & 63) == 0 {
                cur_bits = seed.next_u64();
            }
            let bit = (cur_bits & 1) == 1;
            cur_bits >>= 1;
            bit
        }),
    )
}

/// The hash_key function output a row in the sparse matrix.
/// The row is a band of length 'w' of 0/1 values.
pub fn hash_key<const W: usize, K: OkvsKey>(
    k: &K,
    m: usize,
    base_seed: &[u8; 16],
) -> (usize, [u64; W]) {
    let (band_start, mut band_iter) = hash_key_compressed::<W, _>(k, m, base_seed);
    let mut output = [0u64; W];
    for (idx, b) in band_iter.enumerate() {
        if !b {
            continue;
        }
        let cell = idx >> 6;
        let offset = idx & 63;
        output[cell] ^= 1u64 << offset;
    }
    (band_start, output)
}
pub fn encode<const W: usize, K: OkvsKey, V: BinaryOkvsValue>(
    kvs: &[(K, V)],
    epsilon_percent: EpsilonPercent,
) -> BinaryEncodedOkvs<W, K, V> {
    let mut base_seed = [0u8; 16];
    thread_rng().fill_bytes(&mut base_seed);
    let epsilon_percent_usize = usize::from(epsilon_percent);
    let m = ((100 + epsilon_percent_usize) * kvs.len())
        .div_ceil(100)
        .max(W * 64 + 1);
    let mut matrix: Vec<_> = kvs
        .iter()
        .map(|(k, v)| BinaryMatrixRow::<W, V>::from_key_value(k, *v, m, &base_seed))
        .collect();
    matrix.sort_by_key(|f| f.first_col);
    // Now we start the Gaussian Elimination.
    // we iterate on the rows
    for i in 0..matrix.len() - 1 {
        let start_column = matrix[i].first_col;
        let row_ref = unsafe { &*(&matrix[i] as *const BinaryMatrixRow<W, V>) };
        // for each row we subtract it from all the rows that contain it.
        let mut j = i + 1;
        let mut max_col = start_column;
        while j < matrix.len() {
            // If same rows has first non-zero column equal, subtract them.
            if matrix[j].first_col == start_column {
                matrix[j] -= row_ref;
                debug_assert!(matrix[j].first_col > start_column);
                max_col = matrix[j].first_col.max(max_col);
                j += 1;
            } else {
                break;
            }
        }
        let mut max_index = j;
        while max_index < matrix.len() && matrix[max_index].first_col < max_col {
            max_index += 1;
        }
        matrix[i + 1..max_index].sort_by_key(|v| v.first_col);
        // we re-sort the rows according to first non-zero column.
    }

    let mut output_vector = vec![None; m];
    // start with back substitution, we start with last line.
    for i in (0..matrix.len()).rev() {
        let current_first_col = matrix[i].first_col;
        let rhs = matrix[i].rhs;
        debug_assert!(output_vector[current_first_col].is_none());
        output_vector[current_first_col] = Some(rhs);
        if i > 0 {
            let mut j = i - 1;
            while matrix[j].first_col + 64 * W > current_first_col {
                if matrix[j].get_bit_value_uncheck(current_first_col) {
                    matrix[j].rhs ^= rhs;
                }
                if j > 0 {
                    j -= 1;
                } else {
                    break;
                }
            }
        }
    }
    let v = output_vector
        .into_iter()
        .map(|v| {
            if v.is_none() {
                V::default()
            } else {
                v.unwrap()
            }
        })
        .collect();
    return BinaryEncodedOkvs(v, PhantomData, base_seed);
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use super::{encode, BinaryOkvsValue};
    use crate::rb_okvs::{EpsilonPercent, OkvsKey, OkvsU128};

    fn test_okvs<const W: usize, K: OkvsKey, V: BinaryOkvsValue>(
        kvs: &[(K, V)],
        epsilon_percent: EpsilonPercent,
    ) {
        let encoded = encode::<W, K, V>(&kvs, epsilon_percent);
        for (k, v) in kvs {
            assert_eq!(encoded.decode(k), *v)
        }
    }
    fn randomize_kvs<K: OkvsKey, V: BinaryOkvsValue>(size: usize) -> Vec<(K, V)> {
        let mut rng = thread_rng();
        (0..size)
            .map(|_| (K::random(&mut rng), V::random(&mut rng)))
            .collect()
    }
    #[test]
    fn test_okvs_small() {
        let kvs = randomize_kvs(300);
        test_okvs::<7, OkvsU128, OkvsU128>(&kvs, EpsilonPercent::Three);
        test_okvs::<5, OkvsU128, OkvsU128>(&kvs, EpsilonPercent::Five);
        test_okvs::<4, OkvsU128, OkvsU128>(&kvs, EpsilonPercent::Seven);
        test_okvs::<4, OkvsU128, OkvsU128>(&kvs, EpsilonPercent::Ten);
    }
}
