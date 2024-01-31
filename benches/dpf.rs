use std::collections::{HashMap, HashSet};

use criterion::{criterion_group, criterion_main, Criterion};
use dmpf::{Dmpf, DmpfKey, DpfDmpf, DpfKey, DpfOutput, Node, PrimeField64, PrimeField64x2};
use rand::{thread_rng, RngCore};

const INPUT_LENS: (usize, usize) = (5, 20);
const POINTS: [usize; 3] = [5 * 5, 14 * 14, 70 * 70];

fn make_inputs<F: DpfOutput>(input_len: usize, total_inputs: usize) -> Vec<(u128, F)> {
    let domain_size = 1 << input_len;
    let mut rng = thread_rng();
    let mut map = HashMap::<u128, F>::with_capacity(total_inputs);
    while map.len() < total_inputs {
        let i = rng.next_u64() % domain_size;
        let i_encoded = (i as u128) << (128 - input_len);
        if map.contains_key(&i_encoded) {
            continue;
        }
        let f = F::random(&mut rng);
        map.insert(i_encoded, f);
    }
    let mut output: Vec<_> = map.into_iter().collect();
    output.sort_unstable_by_key(|f| f.0);
    output
}

fn bench_dmpf<F: DpfOutput, D: Dmpf<F>>(
    c: &mut Criterion,
    id: &str,
    d: &D,
    input_lens: (usize, usize),
    points: Vec<usize>,
) {
    let mut rng = thread_rng();
    for input_len in input_lens.0..input_lens.1 {
        for points_count in points.iter().copied() {
            if points_count > (1 << (input_len - 1)) {
                continue;
            }
            let inputs = make_inputs::<F>(input_len, points_count);
            // Generating the key
            c.bench_with_input(
                criterion::BenchmarkId::new(
                    format!("{}/Keygen", id),
                    format!("{},{}", input_len.to_string(), points_count),
                ),
                &(input_len, inputs.clone()),
                |b, input| {
                    let input_len = input.0;
                    let inputs = &input.1;
                    b.iter(|| {
                        d.try_gen(input_len, &inputs, &mut rng).unwrap();
                    })
                },
            );
            c.bench_with_input(
                criterion::BenchmarkId::new(
                    format!("{}/EvalSingle", id),
                    format!("{},{}", input_len.to_string(), points_count),
                ),
                &(input_len, inputs.clone()),
                |b, input| {
                    let input_len = input.0;
                    let inputs = &input.1;
                    let (k_0, _) = d.try_gen(input_len, &inputs, &mut rng).unwrap();
                    let random_point = rng.next_u64() % (1 << input_len);
                    let random_point_encoded = (random_point as u128) << (128 - input_len);
                    let mut f = F::default();
                    b.iter(|| {
                        k_0.eval(&random_point_encoded, &mut f);
                    })
                },
            );
            c.bench_with_input(
                criterion::BenchmarkId::new(
                    format!("{}/EvalAll", id),
                    format!("{},{}", input_len.to_string(), points_count),
                ),
                &(input_len, inputs),
                |b, input| {
                    let input_len = input.0;
                    let inputs = &input.1;
                    let (k_0, _) = d.try_gen(input_len, &inputs, &mut rng).unwrap();
                    b.iter(|| {
                        k_0.eval_all();
                    })
                },
            );
        }
    }
}

fn bench_dpf_dmpf(c: &mut Criterion) {
    let dpf = DpfDmpf::new();
    let points = POINTS.to_vec();
    bench_dmpf::<PrimeField64x2, _>(c, "dpf_dmpf", &dpf, INPUT_LENS, points);
}

fn bench_dpf(c: &mut Criterion) {
    let mut rng = thread_rng();
    for input_len in INPUT_LENS.0..INPUT_LENS.1 {
        let roots = (Node::random(&mut rng), Node::random(&mut rng));
        let alpha = ((rng.next_u64() % (1 << input_len)) as u128) << (128 - input_len);
        let beta = PrimeField64x2::random(&mut rng);
        c.bench_with_input(
            criterion::BenchmarkId::new(
                format!("Dpf/KeyGen"),
                format!("{}", input_len.to_string()),
            ),
            &input_len,
            |b, input_len| {
                b.iter(|| DpfKey::gen(&roots, &alpha, *input_len, &beta));
            },
        );
        c.bench_with_input(
            criterion::BenchmarkId::new(
                format!("Dpf/EvalSingle"),
                format!("{}", input_len.to_string()),
            ),
            &input_len,
            |b, input_len| {
                let (k_0, _) = DpfKey::gen(&roots, &alpha, *input_len, &beta);
                let mut output = PrimeField64x2::default();
                let x = ((rng.next_u64() % (1 << input_len)) as u128) << (128 - input_len);
                b.iter(|| k_0.eval(&x, &mut output));
            },
        );
        c.bench_with_input(
            criterion::BenchmarkId::new(
                format!("Dpf/EvalAll"),
                format!("{}", input_len.to_string()),
            ),
            &input_len,
            |b, input_len| {
                let (k_0, _) = DpfKey::gen(&roots, &alpha, *input_len, &beta);
                b.iter(|| k_0.eval_all());
            },
        );
    }
}

criterion_group!(benches, bench_dpf, bench_dpf_dmpf);
criterion_main!(benches);
