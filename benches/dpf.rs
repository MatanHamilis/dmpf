use core::panic;
use std::collections::HashMap;

use criterion::{criterion_group, criterion_main, Criterion};
use dmpf::{
    batch_code::BatchCodeDmpf, big_state::BigStateDmpf, g, okvs::OkvsDmpf, Dmpf, DmpfKey, DpfDmpf,
    DpfKey, DpfOutput, EpsilonPercent, LogN, Node, Node512, PrimeField64x2,
};
use rand::{thread_rng, RngCore};

const INPUT_LENS: [usize; 5] = [10, 12, 14, 16, 18];
const POINTS: [usize; 6] = [9, 10, 11, 12, 13, 14];

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
    for input_len in INPUT_LENS {
        for points in POINTS {
            bench_dmpf::<PrimeField64x2, _>(c, "dpf_dmpf", &dpf, input_len, points);
        }
    }
}
fn bench_dpf_dmpf_node(c: &mut Criterion) {
    let dpf = DpfDmpf::new();
    for input_len in INPUT_LENS {
        for points in POINTS {
            bench_dmpf::<Node, _>(c, "dpf_dmpf_node", &dpf, input_len, points);
        }
    }
}
fn bench_dpf_dmpf_node512(c: &mut Criterion) {
    let dpf = DpfDmpf::new();
    for input_len in INPUT_LENS {
        for points in POINTS {
            bench_dmpf::<Node512, _>(c, "dpf_dmpf_node512", &dpf, input_len, points);
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
fn do_bench_okvs_dmpf<O: DpfOutput>(c: &mut Criterion, name: &str) {
    // const LAMBDA: usize = 6;
    const BATCH_SIZE: usize = 1;
    let eps = EpsilonPercent::Hundred;
    for input_len in INPUT_LENS {
        for points in POINTS {
            // let w = g(LAMBDA, eps, match_logn(points).unwrap());
            let w = 12;
            match w {
                12 => {
                    let dpf = OkvsDmpf::<1, 12, O>::new(eps, BATCH_SIZE);
                    bench_dmpf(c, name, &dpf, input_len, points);
                }
                168 => {
                    let dpf = OkvsDmpf::<3, 168, O>::new(eps, BATCH_SIZE);
                    bench_dmpf(c, name, &dpf, input_len, points);
                }
                183 => {
                    let dpf = OkvsDmpf::<3, 183, O>::new(eps, BATCH_SIZE);
                    bench_dmpf(c, name, &dpf, input_len, points);
                }
                _ => panic!("w missing: {}", w),
            }
        }
    }
}
fn bench_okvs_dmpf(c: &mut Criterion) {
    do_bench_okvs_dmpf::<PrimeField64x2>(c, "okvs");
}
fn bench_okvs_dmpf_node(c: &mut Criterion) {
    do_bench_okvs_dmpf::<Node>(c, "okvs_node");
}
fn bench_okvs_dmpf_node512(c: &mut Criterion) {
    do_bench_okvs_dmpf::<Node512>(c, "okvs_node");
}

fn bench_batch_code_dmpf(c: &mut Criterion) {
    let dpf = BatchCodeDmpf::<PrimeField64x2>::new();
    for input_len in INPUT_LENS {
        for points in POINTS {
            bench_dmpf::<PrimeField64x2, _>(c, "batch_code", &dpf, input_len, points);
        }
    }
}
fn bench_batch_code_dmpf_node(c: &mut Criterion) {
    let dpf = BatchCodeDmpf::<Node>::new();
    for input_len in INPUT_LENS {
        for points in POINTS {
            bench_dmpf::<Node, _>(c, "batch_code_node", &dpf, input_len, points);
        }
    }
}
// fn bench_batch_code_dmpf_node512(c: &mut Criterion) {
//     let dpf = BatchCodeDmpf::<Node>::new();
//     for input_len in INPUT_LENS {
//         for points in POINTS {
//             bench_dmpf::<Node512, _>(c, "batch_code_node512", &dpf, input_len, points);
//         }
//     }
// }

fn bench_dpf_single(c: &mut Criterion) {
    let mut rng = thread_rng();
    for input_len in INPUT_LENS {
        let roots = (Node::random(&mut rng), Node::random(&mut rng));
        let alpha = ((rng.next_u64() % (1 << input_len)) as u128) << (128 - input_len);
        // let beta = PrimeField64x2::random(&mut rng);
        let beta = Node::random(&mut rng);
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
                let mut output = Node::default();
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
    let dpf = BigStateDmpf::new(8);
    for input_len in INPUT_LENS {
        for points in POINTS {
            bench_dmpf::<PrimeField64x2, _>(c, "big_state", &dpf, input_len, points);
        }
    }
}

fn bench_big_state_dmpf_node(c: &mut Criterion) {
    let dpf = BigStateDmpf::new(8);
    for input_len in INPUT_LENS {
        for points in POINTS {
            bench_dmpf::<Node, _>(c, "big_state_node", &dpf, input_len, points);
        }
    }
}
fn bench_big_state_dmpf_node512(c: &mut Criterion) {
    let dpf = BigStateDmpf::new(8);
    for input_len in INPUT_LENS {
        for points in POINTS {
            bench_dmpf::<Node512, _>(c, "big_state_node512", &dpf, input_len, points);
        }
    }
}
criterion_group!(
    name = benches;
    config = Criterion::default().configure_from_args();
    targets = bench_dpf_single,
    bench_dpf_dmpf,
    bench_dpf_dmpf_node,
    bench_dpf_dmpf_node512,
    bench_big_state_dmpf,
    bench_big_state_dmpf_node,
    bench_big_state_dmpf_node512,
    bench_batch_code_dmpf,
    bench_batch_code_dmpf_node,
    // bench_batch_code_dmpf_node512,
    bench_okvs_dmpf,
    bench_okvs_dmpf_node,
    bench_okvs_dmpf_node512
);
criterion_main!(benches);
