use aes_prng::AesRng;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dmpf::batch_code::{expansion_param_from_points, gen_hash_functions};
use rand::{thread_rng, RngCore};

fn bench_batch_code(c: &mut Criterion) {
    const INPUT_SIZE: (usize, usize) = (5, 20);
    const NON_ZERO_POINTS: [usize; 3] = [5 * 5, 14 * 14, 66 * 66];
    const HASH_FUNCTIONS_COUNT: usize = 3;
    for log_input_size in INPUT_SIZE.0..=INPUT_SIZE.1 {
        for non_zero_points in NON_ZERO_POINTS {
            let expansion_overhead_in_percent = expansion_param_from_points(non_zero_points);
            let buckets = (non_zero_points * (100 + expansion_overhead_in_percent)) / 100;
            let input_domain_size = 1 << log_input_size;
            let bucket_size = (input_domain_size * HASH_FUNCTIONS_COUNT).div_ceil(buckets);
            c.bench_with_input(
                BenchmarkId::new(
                    "gen_hash_functions",
                    format!(
                        "log_input_size:{}/non_zero_points:{}",
                        log_input_size, non_zero_points
                    ),
                ),
                &(log_input_size, bucket_size),
                |b, input| {
                    let log_input_size = input.0;
                    let bucket_size = input.1;
                    let mut seed = [0u8; 16];
                    thread_rng().fill_bytes(&mut seed);
                    b.iter(|| gen_hash_functions::<AesRng>(log_input_size, bucket_size, 3, seed))
                },
            );
        }
    }
}

criterion_group!(benches, bench_batch_code);
criterion_main!(benches);
