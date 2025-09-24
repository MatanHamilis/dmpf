//! In this module we benchmark the FFT implementation.

use aes_prng::AesRng;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dmpf::field::FieldElement;
use std::env;

pub fn fft_benchmark(c: &mut Criterion) {
    let key = "FFT_BENCH_MIN_LOG_SIZE";
    let min_log_size = match env::var(key) {
        Ok(val) => val.parse().unwrap_or(17),
        Err(_) => 17,
    };
    let key = "FFT_BENCH_MAX_LOG_SIZE";
    let max_log_size = match env::var(key) {
        Ok(val) => val.parse().unwrap_or(25),
        Err(_) => 25,
    };
    if min_log_size > max_log_size || min_log_size < 3 {
        panic!(
            "Invalid FFT bench size range: {min_log_size} to {max_log_size}. Must be at least 3."
        );
    }
    let mut group = c.benchmark_group("FFT");
    for log_size in min_log_size..=max_log_size {
        let size = 1 << log_size;
        let mut rng = AesRng::from_random_seed();
        let coeffs: Vec<_> = (0..size)
            .map(|_| dmpf::field::PrimeField64::random(&mut rng))
            .collect();
        group.bench_function(format!("FFT size {size}"), |b| {
            b.iter(|| {
                let _res = black_box(ole_pcg::fft::fft(&coeffs));
            })
        });
    }
    group.finish();
}
criterion_group!(benches, fft_benchmark);
criterion_main!(benches);
