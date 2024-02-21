use core::panic;
use std::collections::HashMap;

use criterion::{criterion_group, criterion_main, Criterion};
use dmpf::{
    batch_code::BatchCodeDmpf, big_state::BigStateDmpf, g, okvs::OkvsDmpf, Dmpf, DmpfKey, DpfDmpf,
    DpfKey, DpfOutput, EpsilonPercent, LogN, Node, PrimeField64x2,
};
use rand::{thread_rng, RngCore};

const INPUT_LENS: (usize, usize) = (5, 30);
const POINTS: [usize; 4] = [10, 5 * 5, 14 * 14, 66 * 66];

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
    input_len: usize,
    points: usize,
) {
    let mut rng = thread_rng();
    if points > (1 << (input_len - 1)) {
        return;
    }
    let inputs = make_inputs::<F>(input_len, points);
    // Generating the key
    c.bench_with_input(
        criterion::BenchmarkId::new(
            format!("{}/Keygen", id),
            format!("{},{}", input_len.to_string(), points),
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
            format!("{},{}", input_len.to_string(), points),
        ),
        &(input_len, inputs.clone()),
        |b, input| {
            let input_len = input.0;
            let inputs = &input.1;
            let (k_0, _) = d.try_gen(input_len, &inputs, &mut rng).unwrap();
            let random_point = rng.next_u64() % (1 << input_len);
            let random_point_encoded = (random_point as u128) << (128 - input_len);
            let mut f = F::default();
            let mut session = k_0.make_session();
            k_0.eval_with_session(&random_point_encoded, &mut f, &mut session);
            b.iter(|| {
                k_0.eval_with_session(&random_point_encoded, &mut f, &mut session);
            })
        },
    );
    c.bench_with_input(
        criterion::BenchmarkId::new(
            format!("{}/EvalAll", id),
            format!("{},{}", input_len.to_string(), points),
        ),
        &(input_len, inputs),
        |b, input| {
            b.iter_batched_ref(
                || {
                    let input_len = input.0;
                    let inputs = &input.1;
                    let (k_0, _) = d.try_gen(input_len, &inputs, &mut rng).unwrap();
                    k_0
                },
                |k_0| {
                    k_0.eval_all();
                },
                criterion::BatchSize::NumBatches(2),
            )
        },
    );
}

fn bench_dpf_dmpf(c: &mut Criterion) {
    let dpf = DpfDmpf::new();
    for input_len in INPUT_LENS.0..=INPUT_LENS.1 {
        for points in POINTS {
            bench_dmpf::<PrimeField64x2, _>(c, "dpf_dmpf", &dpf, input_len, points);
        }
    }
}
fn match_logn(points: usize) -> Option<LogN> {
    Some(if points < 1 << 10 {
        LogN::Ten
    } else if points < 1 << 14 {
        LogN::Fourteen
    } else if points < 1 << 16 {
        LogN::Sixteen
    } else if points < 1 << 18 {
        LogN::Eighteen
    } else if points < 1 << 20 {
        LogN::Twenty
    } else if points < 1 << 24 {
        LogN::TwentyFour
    } else {
        return None;
    })
}
fn bench_okvs_dmpf(c: &mut Criterion) {
    const LAMBDA: usize = 40;
    let eps = EpsilonPercent::Fifty;
    for input_len in INPUT_LENS.0..=INPUT_LENS.1 {
        for points in POINTS {
            let w = g(LAMBDA, eps, match_logn(points).unwrap());
            match w {
                16 => {
                    let dpf = OkvsDmpf::<1, 40, PrimeField64x2>::new(eps);
                    bench_dmpf(c, "okvs", &dpf, input_len, points);
                }
                18 => {
                    let dpf = OkvsDmpf::<1, 40, PrimeField64x2>::new(eps);
                    bench_dmpf(c, "okvs", &dpf, input_len, points);
                }
                33 => {
                    let dpf = OkvsDmpf::<1, 40, PrimeField64x2>::new(eps);
                    bench_dmpf(c, "okvs", &dpf, input_len, points);
                }
                36 => {
                    let dpf = OkvsDmpf::<1, 40, PrimeField64x2>::new(eps);
                    bench_dmpf(c, "okvs", &dpf, input_len, points);
                }
                168 => {
                    let dpf = OkvsDmpf::<3, 168, PrimeField64x2>::new(eps);
                    bench_dmpf(c, "okvs", &dpf, input_len, points);
                }
                183 => {
                    let dpf = OkvsDmpf::<3, 183, PrimeField64x2>::new(eps);
                    bench_dmpf(c, "okvs", &dpf, input_len, points);
                }
                _ => panic!("w missing: {}", w),
            }
        }
    }
}

fn bench_batch_code_dmpf(c: &mut Criterion) {
    let dpf = BatchCodeDmpf::<PrimeField64x2>::new();
    for input_len in INPUT_LENS.0..=INPUT_LENS.1 {
        for points in POINTS {
            bench_dmpf::<PrimeField64x2, _>(c, "batch_code", &dpf, input_len, points);
        }
    }
}

fn bench_dpf_single(c: &mut Criterion) {
    let mut rng = thread_rng();
    for input_len in INPUT_LENS.0..=INPUT_LENS.1 {
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

fn bench_big_state_dmpf(c: &mut Criterion) {
    let dpf = BigStateDmpf::new(4);
    for input_len in INPUT_LENS.0..=INPUT_LENS.1 {
        for points in POINTS {
            bench_dmpf::<PrimeField64x2, _>(c, "big_state", &dpf, input_len, points);
        }
    }
}

fn bench_big_state_dmpf_node(c: &mut Criterion) {
    let dpf = BigStateDmpf::new(4);
    for input_len in INPUT_LENS.0..=INPUT_LENS.1 {
        for points in POINTS {
            bench_dmpf::<Node, _>(c, "big_state_node", &dpf, input_len, points);
        }
    }
}
criterion_group!(
    name = benches;
    config = Criterion::default().configure_from_args();
    targets = bench_dpf_single,
    bench_dpf_dmpf,
    bench_big_state_dmpf,
    bench_big_state_dmpf_node,
    bench_batch_code_dmpf,
    bench_okvs_dmpf,
);
criterion_main!(benches);
