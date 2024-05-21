pub mod binary_okvs;
use std::{
    fmt::Debug,
    hash::Hash,
    iter::Sum,
    marker::PhantomData,
    ops::{Add, AddAssign, Div, Mul, MulAssign, SubAssign},
};

use aes_prng::AesRng;
use rand::{thread_rng, CryptoRng, RngCore, SeedableRng};

pub trait OkvsValue:
    Default
    + Clone
    + Copy
    + Mul<bool, Output = Self>
    + Mul<Output = Self>
    + MulAssign
    + Eq
    + PartialEq
    + Debug
    + Hash
    + AddAssign
    + Add<Output = Self>
    + SubAssign
    + Div<Output = Self>
    + From<bool>
    + Sum
{
    fn random<R: CryptoRng + RngCore>(rng: R) -> Self;
    fn is_zero(&self) -> bool;
    fn inv(&self) -> Self;
}

pub trait OkvsKey {
    fn hash_seed(&self, base_seed: &[u8; 16]) -> [u8; 16];
    fn random<R: CryptoRng + RngCore>(rng: R) -> Self;
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
pub struct OkvsU128(u128);
impl OkvsKey for OkvsU128 {
    fn hash_seed(&self, base_seed: &[u8; 16]) -> [u8; 16] {
        let my_bytes = self.0.to_le_bytes();
        core::array::from_fn(|i| my_bytes[i] ^ base_seed[i])
    }
    fn random<R: CryptoRng + RngCore>(mut rng: R) -> Self {
        let mut bytes = [0u8; 16];
        rng.fill_bytes(&mut bytes);
        Self(u128::from_be_bytes(bytes))
    }
}
impl OkvsU128 {
    pub fn new(v: u128, input_size: usize) -> Self {
        Self(v << (128 - input_size))
    }
}

/// This type is mainly used for testing.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub struct OkvsBool(bool);
impl Add for OkvsBool {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        self.0 ^= rhs.0;
        self
    }
}
impl OkvsValue for OkvsBool {
    fn is_zero(&self) -> bool {
        !self.0
    }
    fn random<R: CryptoRng + RngCore>(mut rng: R) -> Self {
        Self(rng.next_u32() & 1 == 1)
    }
    fn inv(&self) -> Self {
        assert!(!self.is_zero());
        self.clone()
    }
}
impl MulAssign for OkvsBool {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}
impl Mul for OkvsBool {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}
impl Default for OkvsBool {
    fn default() -> Self {
        Self(false)
    }
}
impl Mul<bool> for OkvsBool {
    type Output = Self;
    fn mul(self, rhs: bool) -> Self::Output {
        Self(self.0 & rhs)
    }
}
impl Sum for OkvsBool {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut sum = false;
        iter.for_each(|v| sum ^= v.0);
        Self(sum)
    }
}
impl From<bool> for OkvsBool {
    fn from(value: bool) -> Self {
        Self(value)
    }
}
impl Div for OkvsBool {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        assert_ne!(rhs.0, false);
        self
    }
}
impl SubAssign for OkvsBool {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0
    }
}
impl AddAssign for OkvsBool {
    fn add_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0
    }
}

#[derive(Debug, Clone, Copy)]
struct MatrixRow<const W: usize, V: OkvsValue> {
    first_col: usize,
    band: [V; W],
    rhs: V,
}

impl<const W: usize, V: OkvsValue> MatrixRow<W, V> {
    // Invariant: We always subtract only aligned rows (i.e. for which the band starts at the same column).
    fn try_sub_assign(&mut self, rhs: &Self) -> Option<()> {
        // If I'm being subtracted, then anyway nothing should change *before* my first column.
        debug_assert_eq!(self.first_col, rhs.first_col);
        let t = rhs.get_bit_value_uncheck(self.first_col);
        assert_eq!(t / t, t);
        let factor = self.band[0] / rhs.get_bit_value_uncheck(self.first_col);
        assert!(!factor.is_zero());
        // println!("Before, factor: {factor:?}");
        // dbg!(&self.band);
        self.band
            .iter_mut()
            .zip(rhs.band.iter())
            .for_each(|(a, b)| *a -= *b * factor);
        self.rhs -= rhs.rhs * factor;
        // println!("After subtract");
        // dbg!(&self.band);
        self.align()?;
        // println!("After align");
        // dbg!(&self.band);
        self.normalize();
        // println!("After normalize");
        // dbg!(&self.band);
        Some(())
    }
    fn from_key_value<K: OkvsKey>(
        key: &K,
        value: V,
        m: usize,
        batch_size: usize,
        base_seed: &[u8; 16],
    ) -> Self {
        let (first_col, band) = hash_key(key, m, batch_size, base_seed);
        let mut output = Self {
            band,
            first_col,
            rhs: value,
        };
        output.align();
        output
    }
    fn normalize(&mut self) {
        let factor = self.band[0].inv();
        self.band.iter_mut().for_each(|v| *v *= factor);
        self.rhs *= factor;
    }
    fn align(&mut self) -> Option<()> {
        // Safety: there will be no zeros with negligible probability!
        let shift = self.band.iter().enumerate().find(|(_, v)| !v.is_zero())?.0;
        for i in 0..W - shift {
            self.band[i] = self.band[i + shift];
        }
        for i in W - shift..W {
            self.band[i] = V::default();
        }

        self.first_col += shift as usize;
        Some(())
    }
    fn get_bit_value_uncheck(&self, column: usize) -> V {
        self.band[column - self.first_col]
    }
}

// This function determines from epsilon (the encoding overhead), Lambda (the statistical security parameter)
// and n (the number of samples) the value of w (rounded to a multiple of 64).
#[derive(Clone, Copy)]
pub enum EpsilonPercent {
    Three = 0,
    Five,
    Seven,
    Ten,
    Fifty,
    Hundred,
}

impl From<EpsilonPercent> for usize {
    fn from(value: EpsilonPercent) -> Self {
        match value {
            EpsilonPercent::Three => 3,
            EpsilonPercent::Five => 5,
            EpsilonPercent::Seven => 7,
            EpsilonPercent::Ten => 10,
            EpsilonPercent::Fifty => 50,
            EpsilonPercent::Hundred => 100,
        }
    }
}

impl TryFrom<usize> for EpsilonPercent {
    type Error = ();
    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(match value {
            3 => EpsilonPercent::Three,
            5 => EpsilonPercent::Five,
            7 => EpsilonPercent::Seven,
            10 => EpsilonPercent::Ten,
            50 => EpsilonPercent::Fifty,
            100 => EpsilonPercent::Hundred,
            _ => return Err(()),
        })
    }
}

#[derive(Clone, Copy)]
pub enum LogN {
    Ten = 0,
    Fourteen,
    Sixteen,
    Eighteen,
    Twenty,
    TwentyFour,
}

impl TryFrom<usize> for LogN {
    type Error = ();
    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(match value {
            10 => Self::Ten,
            14 => Self::Fourteen,
            16 => Self::Sixteen,
            18 => Self::Eighteen,
            20 => Self::Twenty,
            24 => Self::TwentyFour,
            _ => return Err(()),
        })
    }
}

impl From<LogN> for usize {
    fn from(value: LogN) -> Self {
        match value {
            LogN::Ten => 10,
            LogN::Fourteen => 14,
            LogN::Sixteen => 16,
            LogN::Eighteen => 18,
            LogN::Twenty => 20,
            LogN::TwentyFour => 24,
        }
    }
}

// From RB-OKVS paper appendix F.
// Essentially for LAMBDA = 40, W ranges between 64-bit array of size 4 to 9.
// For "Else" we follow theorem 1 from the RB-OKVS paper and show that
pub const fn g(lambda: usize, epsilon_percent: EpsilonPercent, log_n: LogN) -> usize {
    let a_b_tables = [
        //epsilon = 3
        [
            (8047, 3464),
            (8253, 5751),
            (8241, 7023),
            (8192, 8569),
            (8313, 10880),
            (8253, 14671),
        ],
        //epsilon = 5
        [
            (13880, 4424),
            (13890, 6976),
            (13990, 8942),
            (13880, 10710),
            (14070, 12920),
            (13760, 16741),
        ],
        //epsilon = 7
        [
            (19470, 5383),
            (19260, 8150),
            (19610, 10430),
            (19550, 12300),
            (19390, 14100),
            (13760, 16741), // Item missing in the paper! Duplicate previous
        ],
        // epsilon = 10
        [
            (27470, 6296),
            (26850, 9339),
            (27400, 11610),
            (27150, 13390),
            (26910, 15210),
            (27510, 19830),
        ],
        // epsilon = 50
        [
            (27470 * 5, 6296),
            (26850 * 5, 9339),
            (27400 * 5, 11610),
            (27150 * 5, 13390),
            (26910 * 5, 15210),
            (27510 * 5, 19830),
        ],
        // epsilon = 100
        [
            (27470 * 10, 6296),
            (26850 * 10, 9339),
            (27400 * 10, 11610),
            (27150 * 10, 13390),
            (26910 * 10, 15210),
            (27510 * 10, 19830),
        ],
    ];
    let (a, b) = a_b_tables[epsilon_percent as usize][log_n as usize];
    // ((100_000 * lambda + 100 * b) / a).div_ceil(64)
    let t = (100_000 * lambda + 100 * b) / a;
    if t > lambda {
        t
    } else {
        lambda
    }
}

#[derive(Clone)]
pub struct EncodedOkvs<const W: usize, K: OkvsKey, V: OkvsValue>(
    Vec<V>,
    PhantomData<K>,
    Vec<Vec<V>>,
    usize,
    [u8; 16],
);
impl<const W: usize, K: OkvsKey, V: OkvsValue> EncodedOkvs<W, K, V> {
    pub fn decode(&self, key: &K) -> V {
        let (offset, bits) = hash_key_compressed_batch::<W, K>(key, self.0.len(), self.3, &self.4);
        let rounded_offset = offset / self.3;
        let slice = &self.2[rounded_offset..];
        bits.zip(slice).map(|(bit, v)| v[bit]).sum()
    }
}

/// We split the functions to avoid generating a large array when `decode`-ing.
pub fn hash_key_compressed<const W: usize, K: OkvsKey>(
    k: &K,
    m: usize,
    batch_size: usize,
    base_seed: &[u8; 16],
) -> (usize, impl Iterator<Item = bool>) {
    let block = k.hash_seed(base_seed);
    let mut seed = AesRng::from_seed(block);
    let mut cur_bits = 0;
    let band_start = (seed.next_u64() as usize) % (m - W);
    let band_start_rounded = (band_start / batch_size) * batch_size;
    let diff = band_start - band_start_rounded;
    (
        band_start,
        std::iter::once(true).chain(
            (0..W + diff)
                .map(move |i| {
                    if (i & 63) == 0 {
                        cur_bits = seed.next_u64() as usize;
                    }
                    let bit = (cur_bits & 1) == 1;
                    cur_bits >>= 1;
                    bit
                })
                .skip(diff + 1),
        ),
    )
}
/// We split the functions to avoid generating a large array when `decode`-ing.
pub fn hash_key_compressed_batch<const W: usize, K: OkvsKey>(
    k: &K,
    m: usize,
    batch_size: usize,
    base_seed: &[u8; 16],
) -> (usize, impl Iterator<Item = usize>) {
    let block = k.hash_seed(base_seed);
    let mut seed = AesRng::from_seed(block);
    let mut cur_bits = 0;
    let band_start = (seed.next_u64() as usize) % (m - W);
    // the band start is rounded to a multiple of batch_size, extra places will be given zeros.
    let band_start_rounded = (band_start / batch_size) * batch_size;
    let diff = band_start - band_start_rounded;
    let mut to_generate = W + diff;
    let mut first = true;
    let mask_iter = (64 / batch_size) - 1;
    (
        band_start,
        (0..(to_generate).div_ceil(batch_size)).map(move |i| {
            if (i & mask_iter) == 0 {
                cur_bits = seed.next_u64() as usize;
            }
            let to_gen_now = (to_generate).min(batch_size);
            let mask = (1 << to_gen_now) - 1;
            let mut bit = cur_bits & mask;
            to_generate -= to_gen_now;
            cur_bits >>= batch_size;
            if first {
                first = false;
                bit = ((bit >> diff) | 1) << diff;
            }
            bit
        }),
    )
}

/// The hash_key function output a row in the sparse matrix.
/// The row is a band of length 'w' of 0/1 values.
pub fn hash_key<const W: usize, K: OkvsKey, V: OkvsValue>(
    k: &K,
    m: usize,
    batch_size: usize,
    base_seed: &[u8; 16],
) -> (usize, [V; W]) {
    let (band_start, mut band_iter) = hash_key_compressed::<W, _>(k, m, batch_size, base_seed);
    (
        band_start,
        core::array::from_fn(|_| V::from(band_iter.next().unwrap())),
    )
}
pub fn encode<const W: usize, K: OkvsKey, V: OkvsValue>(
    kvs: &[(K, V)],
    epsilon_percent: EpsilonPercent,
    batch_size: usize,
) -> EncodedOkvs<W, K, V> {
    let mut seed_bytes = [0u8; 16];
    'outer: loop {
        thread_rng().fill_bytes(&mut seed_bytes);
        let epsilon_percent_usize = usize::from(epsilon_percent);
        let m = ((100 + epsilon_percent_usize) * kvs.len())
            .div_ceil(100)
            .max(W + 1);
        let mut matrix: Vec<_> = kvs
            .iter()
            .map(|(k, v)| MatrixRow::<W, V>::from_key_value(k, *v, m, batch_size, &seed_bytes))
            .collect();
        matrix.sort_by_key(|f| f.first_col);
        // Now we start the Gaussian Elimination.
        // we iterate on the rows
        for i in 0..matrix.len() - 1 {
            let start_column = matrix[i].first_col;
            let row_ref = unsafe { &*(&matrix[i] as *const MatrixRow<W, V>) };
            // for each row we subtract it from all the rows that contain it.
            let mut j = i + 1;
            let mut max_col = start_column;
            while j < matrix.len() {
                // If same rows has first non-zero column equal, subtract them.
                if matrix[j].first_col == start_column {
                    if matrix[j].try_sub_assign(row_ref).is_none() {
                        continue 'outer;
                    };
                    debug_assert!(matrix[j].first_col > start_column);
                    debug_assert_eq!(matrix[j].band[0].inv(), matrix[j].band[0]);
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
                while matrix[j].first_col + W > current_first_col {
                    let col_value = matrix[j].get_bit_value_uncheck(current_first_col);
                    if !col_value.is_zero() {
                        matrix[j].rhs -= col_value * rhs;
                    }
                    if j > 0 {
                        j -= 1;
                    } else {
                        break;
                    }
                }
            }
        }
        let v: Vec<_> = output_vector
            .into_iter()
            .map(|v| {
                if v.is_none() {
                    V::default()
                } else {
                    v.unwrap()
                }
            })
            .collect();

        let vv = v
            .chunks(batch_size)
            .map(|oc| {
                let mut cur_v = Vec::with_capacity(1 << oc.len());
                cur_v.push(V::default());
                for val in oc.iter().copied() {
                    for i in 0..cur_v.len() {
                        cur_v.push(cur_v[i] + val);
                    }
                }
                cur_v
            })
            .collect();
        return EncodedOkvs(v, PhantomData, vv, batch_size, seed_bytes);
    }
}

#[cfg(test)]
mod tests {
    #[derive(Hash, Debug, PartialEq, Eq, Clone, Copy)]
    pub struct OkvsMod3(u8);
    impl Add for OkvsMod3 {
        type Output = Self;
        fn add(mut self, rhs: Self) -> Self::Output {
            self.0 += rhs.0;
            self.0 %= 3;
            self
        }
    }
    impl OkvsValue for OkvsMod3 {
        fn is_zero(&self) -> bool {
            self.0 == 0
        }
        fn random<R: rand::prelude::CryptoRng + rand::prelude::RngCore>(mut rng: R) -> Self {
            Self((rng.next_u32() % 3) as u8)
        }
        fn inv(&self) -> Self {
            assert_ne!(self.0, 0);
            self.clone()
        }
    }
    impl MulAssign for OkvsMod3 {
        fn mul_assign(&mut self, rhs: Self) {
            self.0 *= rhs.0;
            self.0 %= 3;
        }
    }
    impl Default for OkvsMod3 {
        fn default() -> Self {
            Self(0)
        }
    }
    impl Mul<bool> for OkvsMod3 {
        type Output = Self;
        fn mul(mut self, rhs: bool) -> Self::Output {
            self.0 *= rhs as u8;
            self
        }
    }
    impl Mul for OkvsMod3 {
        type Output = Self;
        fn mul(mut self, rhs: Self) -> Self::Output {
            self.0 *= rhs.0;
            self.0 %= 3;
            self
        }
    }
    impl AddAssign for OkvsMod3 {
        fn add_assign(&mut self, rhs: Self) {
            self.0 += rhs.0;
            self.0 %= 3;
        }
    }
    impl SubAssign for OkvsMod3 {
        fn sub_assign(&mut self, rhs: Self) {
            self.0 += 3 - rhs.0;
            self.0 %= 3;
        }
    }
    impl Div for OkvsMod3 {
        type Output = Self;
        fn div(self, rhs: Self) -> Self::Output {
            assert!(!rhs.is_zero());
            self * rhs
        }
    }
    impl From<bool> for OkvsMod3 {
        fn from(value: bool) -> Self {
            Self(value as u8)
        }
    }
    impl Sum for OkvsMod3 {
        fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
            let mut sum = Self::default();
            iter.for_each(|v| sum += v);
            sum
        }
    }
    use std::{
        iter::Sum,
        ops::{Add, AddAssign, Div, Mul, MulAssign, SubAssign},
    };

    use rand::thread_rng;

    use crate::{
        rb_okvs::{encode, EpsilonPercent, OkvsKey, OkvsU128, OkvsValue},
        PrimeField64x2,
    };

    fn test_okvs<const W: usize, K: OkvsKey, V: OkvsValue>(
        kvs: &[(K, V)],
        epsilon_percent: EpsilonPercent,
        batch_size: usize,
    ) {
        let encoded = encode::<W, K, V>(&kvs, epsilon_percent, batch_size);
        for (k, v) in kvs {
            assert_eq!(encoded.decode(k), *v)
        }
    }
    fn randomize_kvs<K: OkvsKey, V: OkvsValue>(size: usize) -> Vec<(K, V)> {
        let mut rng = thread_rng();
        (0..size)
            .map(|_| (K::random(&mut rng), V::random(&mut rng)))
            .collect()
    }
    #[test]
    fn test_okvs_small() {
        let kvs = randomize_kvs(300);
        const BATCH_SIZE: usize = 8;
        test_okvs::<6, OkvsU128, PrimeField64x2>(&kvs, EpsilonPercent::Hundred, BATCH_SIZE);
        // test_okvs::<576, OkvsU128, PrimeField64x2>(&kvs, EpsilonPercent::Three, BATCH_SIZE);
        // test_okvs::<400, OkvsU128, PrimeField64x2>(&kvs, EpsilonPercent::Five, BATCH_SIZE);
        // test_okvs::<300, OkvsU128, PrimeField64x2>(&kvs, EpsilonPercent::Seven, BATCH_SIZE);
        // test_okvs::<200, OkvsU128, PrimeField64x2>(&kvs, EpsilonPercent::Ten, BATCH_SIZE);
    }
}
