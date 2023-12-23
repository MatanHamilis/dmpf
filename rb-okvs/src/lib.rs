use std::ops::SubAssign;

use aes_prng::AesRng;
use rand::{RngCore, SeedableRng};

type K = u128;
type V = u128;

#[derive(Debug, Clone, Copy)]
struct MatrixRow<const W: usize> {
    first_col: usize,
    band: [u64; W],
    rhs: V,
}
impl<const W: usize> MatrixRow<W> {
    fn from_key_value(key: K, value: V, m: usize) -> Self {
        let (first_col, band) = hash_key(key, m);
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

// This function determines from epsilon (the encoding overhead), Lambda (the statistical security parameter)
// and n (the number of samples) the value of w (rounded to a multiple of 64).
#[derive(Clone, Copy)]
pub enum EpsilonPercent {
    Three = 0,
    Five,
    Seven,
    Ten,
}

impl From<EpsilonPercent> for usize {
    fn from(value: EpsilonPercent) -> Self {
        match value {
            EpsilonPercent::Three => 3,
            EpsilonPercent::Five => 5,
            EpsilonPercent::Seven => 7,
            EpsilonPercent::Ten => 10,
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
    ];
    let (a, b) = a_b_tables[epsilon_percent as usize][log_n as usize];
    ((100_000 * lambda + 100 * b) / a).div_ceil(64)
}

pub struct EncodedOkvs<const W: usize>(Vec<V>);
impl<const W: usize> EncodedOkvs<W> {
    pub fn decode(&self, key: K) -> V {
        let (offset, bits) = hash_key::<W>(key, self.0.len());
        let slice = &self.0[offset..offset + (W * u64::BITS as usize)];
        bits.iter()
            .copied()
            .zip(slice.chunks(u64::BITS as usize))
            .map(|(mut bits, chunk)| {
                let mut sum: V = 0;
                for idx in 0..chunk.len() {
                    let bit = bits & 1 != 0;
                    bits = bits.overflowing_shr(1).0;
                    if bit {
                        sum ^= chunk[idx]
                    }
                }
                sum
            })
            .fold(0, |cur, acc| cur ^ acc)
    }
}
impl<const W: usize> SubAssign<&MatrixRow<W>> for MatrixRow<W> {
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

/// The hash_key function output a row in the sparse matrix.
/// The row is a band of length 'w' of 0/1 values.
pub fn hash_key<const W: usize>(k: K, m: usize) -> (usize, [u64; W]) {
    let block = unsafe { (&k as *const u128 as *const [u8; 16]).as_ref().unwrap() };
    let mut seed = AesRng::from_seed(*block);
    let band: [u64; W] = core::array::from_fn(|_| seed.next_u64());
    let band_start = (seed.next_u64() as usize) % (m - W * 64);
    (band_start, band)
}
pub fn encode<const W: usize>(kvs: &[(K, V)], epsilon_percent: EpsilonPercent) -> EncodedOkvs<W> {
    let epsilon_percent_usize = usize::from(epsilon_percent);
    let m = ((100 + epsilon_percent_usize) * kvs.len()).div_ceil(100);
    let mut matrix: Vec<_> = kvs
        .iter()
        .copied()
        .map(|(k, v)| MatrixRow::<W>::from_key_value(k, v, m))
        .collect();
    matrix.sort_by_key(|f| f.first_col);
    // Now we start the Gaussian Elimination.
    // we iterate on the rows
    for i in 0..matrix.len() - 1 {
        let start_column = matrix[i].first_col;
        let row_ref = unsafe { &*(&matrix[i] as *const MatrixRow<W>) };
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
        .map(|v| if v.is_none() { 0u128 } else { v.unwrap() })
        .collect();
    return EncodedOkvs(v);
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, RngCore};

    use crate::{encode, EpsilonPercent, K, V};

    const LAMBDA_VAL: usize = 40;
    fn test_okvs<const W: usize>(kvs: &[(K, V)], epsilon_percent: EpsilonPercent) {
        let encoded = encode::<W>(&kvs, epsilon_percent);
        for (k, v) in kvs {
            assert_eq!(encoded.decode(*k), *v)
        }
    }
    fn randomize_kvs(size: usize) -> Vec<(K, V)> {
        (0..size)
            .map(|_| {
                (
                    thread_rng().next_u64() as u128,
                    thread_rng().next_u64() as u128,
                )
            })
            .collect()
    }
    #[test]
    fn test_okvs_small() {
        let kvs = randomize_kvs(10_000);
        test_okvs::<9>(&kvs, EpsilonPercent::Three);
        test_okvs::<7>(&kvs, EpsilonPercent::Five);
        test_okvs::<5>(&kvs, EpsilonPercent::Seven);
        test_okvs::<4>(&kvs, EpsilonPercent::Ten);
    }
}
