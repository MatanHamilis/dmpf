use std::time::Instant;

use dmpf::field::{FieldElement, PowersIterator, RadixTwoFftFriendFieldElement};

fn reverse_bit_orderings<T: Copy + Clone>(v: &[T]) -> Vec<T> {
    let log_size = v.len().ilog2();
    assert_eq!(1 << log_size, v.len());
    (0..v.len()).map(|i| v[reverse_bits(i, log_size)]).collect()
}
pub fn make_generator_powers<F: FieldElement>(generator: F, log_size: u32) -> Vec<F> {
    reverse_bit_orderings(
        PowersIterator::new(generator)
            .take(1 << log_size)
            .collect::<Vec<_>>()
            .as_ref(),
    )
}

fn reverse_bits(value: usize, bits: u32) -> usize {
    (value << (usize::BITS - bits)).reverse_bits()
}

// The generator is generating a multiplicative subgroup of order "coefficients.len()".
// This function essentially treats the coefficients as a polynomial and evaluates it at the powers of generator.
pub fn fft<F: RadixTwoFftFriendFieldElement>(coefficients: &[F]) -> Vec<F> {
    let log_size: u32 = coefficients.len().ilog2();
    let generator = F::generator_for_fft(log_size as usize);
    internal_fft(coefficients, generator)
}
fn internal_fft<F: RadixTwoFftFriendFieldElement>(coefficients: &[F], mut generator: F) -> Vec<F> {
    let time = Instant::now();
    let log_size: u32 = coefficients.len().ilog2();
    assert!(coefficients.len().is_power_of_two());
    let squares: Vec<_> = (0..log_size)
        .map(|_| {
            let o = generator;
            generator = generator.square();
            o
        })
        .collect();
    assert!(generator.is_one());
    println!("FFT First: {}", time.elapsed().as_millis());
    let time = Instant::now();
    let mut output: Vec<F> = reverse_bit_orderings(coefficients);
    println!("FFT Second: {}", time.elapsed().as_millis());
    let time = Instant::now();
    (1..=log_size)
        .zip(squares.iter().rev())
        .for_each(|(log_chk_sz, &g)| {
            let time = Instant::now();
            let chk_sz = 1 << log_chk_sz;
            let powers_time = Instant::now();
            let g_pow: Vec<_> = PowersIterator::new(g).take(chk_sz / 2).collect();
            println!("FFT powers {}", powers_time.elapsed().as_millis());
            let half_len = chk_sz / 2;
            if half_len < 8 {
                output.chunks_mut(chk_sz).for_each(|chk| {
                    for i in 0..half_len {
                        let mid = g_pow[i] * chk[i + half_len];
                        (chk[i], chk[i + half_len]) = (chk[i] + mid, chk[i] - mid);
                    }
                });
            } else {
                output.chunks_mut(chk_sz).for_each(|chk| {
                    for i in 0..half_len / 8 {
                        let four_i = i << 3;
                        let four_i_plus_half_len = half_len + four_i;
                        let mid: [F; 8] = core::array::from_fn(|i| {
                            g_pow[four_i + i] * chk[four_i_plus_half_len + i]
                        });
                        for i in 0..8 {
                            (chk[four_i + i], chk[four_i_plus_half_len + i]) =
                                (chk[four_i + i] + mid[i], chk[four_i + i] - mid[i]);
                        }
                    }
                });
            }
            println!("FFT round {}: {}", log_chk_sz, time.elapsed().as_millis());
        });
    println!("FFT third: {}", time.elapsed().as_millis());
    output
}

pub fn ifft<F: RadixTwoFftFriendFieldElement>(evaluations: &[F]) -> Vec<F> {
    let f_n = F::from(evaluations.len()).inv();
    let log_size: u32 = evaluations.len().ilog2();
    let generator = F::generator_for_fft(log_size as usize);
    let inv_gen = generator.inv();
    let mut output = internal_fft(evaluations, inv_gen);
    output.iter_mut().for_each(|v| *v *= f_n);
    output
}
