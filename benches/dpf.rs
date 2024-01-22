use std::collections::HashMap;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dmpf::{
    batch_code::BatchCodeDmpf,
    big_state::BigStateDpfKey,
    okvs::{Okvs, OkvsDmpf},
    BitSliceMut, BitVec, Dmpf, DmpfKey, DpfKey, Node, BITS_OF_SECURITY,
};
use rand::{thread_rng, RngCore};

const OUTPUT_WIDTH: usize = 128;
pub fn dpf_bench(c: &mut Criterion) {
    let mut rng = thread_rng();
    let root_0 = Node::random(&mut rng);
    let root_1 = Node::random(&mut rng);
    let beta: Vec<bool> = (0..OUTPUT_WIDTH).map(|_| rng.next_u32() & 1 == 1).collect();
    let beta_bitvec = BitVec::from(&beta[..]);
    let roots = (root_0, root_1);
    {
        let mut g = c.benchmark_group("DPF_Keygen");
        for depth in 2..25 {
            let alpha: Vec<bool> = (0..depth).map(|_| rng.next_u32() & 1 == 1).collect();
            let alpha_bv = BitVec::from(&alpha[..]);
            g.bench_with_input(format!("{}", depth), &depth, |b, depth| {
                b.iter(|| DpfKey::gen(&roots, &alpha_bv, &beta_bitvec));
            });
        }
    }
    {
        let mut g = c.benchmark_group("DPF_Eval");
        for depth in 2..25 {
            let alpha: Vec<bool> = (0..depth).map(|_| rng.next_u32() & 1 == 1).collect();
            let alpha_bv = BitVec::from(&alpha[..]);
            g.bench_with_input(format!("{}", depth), &depth, |b, depth| {
                let (k_0, _) = DpfKey::gen(&roots, &alpha_bv, &beta_bitvec);
                let mut x = BitVec::new(*depth);
                for i in 0..*depth {
                    x.set(i, rng.next_u32() & 1 == 1)
                }
                let mut nodes = [Node::default()];
                let mut output = BitSliceMut::new(OUTPUT_WIDTH, &mut nodes);
                b.iter(|| k_0.eval(&(&x).into(), &mut output));
            });
        }
    }
    {
        let mut g = c.benchmark_group("DPF_EvalAll");
        for depth in 2..25 {
            let alpha: Vec<bool> = (0..depth).map(|_| rng.next_u32() & 1 == 1).collect();
            let alpha_bv = BitVec::from(&alpha[..]);
            g.bench_with_input(format!("{}", depth), &depth, |b, depth| {
                let (k_0, _) = DpfKey::gen(&roots, &alpha_bv, &beta_bitvec);
                b.iter(|| k_0.eval_all());
            });
        }
    }
}

pub fn bigstate_dpf_bench(c: &mut Criterion) {
    for points in [5 * 5, 16 * 16, 76 * 76] {
        let mut rng = thread_rng();
        let root_0 = Node::random(&mut rng);
        let root_1 = Node::random(&mut rng);
        let beta: Vec<bool> = (0..OUTPUT_WIDTH).map(|_| rng.next_u32() & 1 == 1).collect();
        let beta_bitvec = BitVec::from(&beta[..]);
        let roots = (root_0, root_1);
        {
            let mut g = c.benchmark_group("BigStateDPF_Keygen");
            for depth in 2..25 {
                if (1 << depth) <= 2 * points {
                    continue;
                }
                let mut ab_map = HashMap::new();
                while ab_map.len() < points {
                    let mut alpha_v = BitVec::new(depth);
                    loop {
                        alpha_v.fill_random(&mut rng);
                        if !ab_map.contains_key(&alpha_v) {
                            break;
                        }
                    }
                    let beta: Vec<bool> =
                        (0..OUTPUT_WIDTH).map(|_| rng.next_u32() & 1 == 1).collect();
                    let beta_bitvec = BitVec::from(&beta[..]);
                    ab_map.insert(alpha_v.clone(), beta_bitvec.clone());
                }
                let alphas_betas: Vec<_> = ab_map.into_iter().collect();
                g.bench_with_input(format!("{}/{}", points, depth), &depth, |b, depth| {
                    b.iter(|| BigStateDpfKey::gen(&alphas_betas, &roots));
                });
            }
        }
        {
            let mut g = c.benchmark_group("BigStateDPF_Eval");
            for depth in 2..25 {
                if (1 << depth) <= 2 * points {
                    continue;
                }
                let mut ab_map = HashMap::new();
                while ab_map.len() < points {
                    let mut alpha_v = BitVec::new(depth);
                    loop {
                        alpha_v.fill_random(&mut rng);
                        if !ab_map.contains_key(&alpha_v) {
                            break;
                        }
                    }
                    let beta: Vec<bool> =
                        (0..OUTPUT_WIDTH).map(|_| rng.next_u32() & 1 == 1).collect();
                    let beta_bitvec = BitVec::from(&beta[..]);
                    ab_map.insert(alpha_v.clone(), beta_bitvec.clone());
                }
                let alphas_betas: Vec<_> = ab_map.into_iter().collect();
                g.bench_with_input(format!("{}/{}", points, depth), &depth, |b, depth| {
                    let (k_0, _) = BigStateDpfKey::gen(&alphas_betas, &roots);
                    let (mut sign_container, mut correction_container, mut expansion_container) =
                        k_0.make_aux_variables();
                    let mut x = BitVec::new(*depth);
                    for i in 0..*depth {
                        x.set(i, rng.next_u32() & 1 == 1)
                    }
                    let mut output = BitVec::new(OUTPUT_WIDTH);
                    b.iter(|| {
                        k_0.eval(
                            &x,
                            &mut output,
                            &mut sign_container,
                            &mut correction_container,
                            &mut expansion_container,
                        )
                    });
                });
            }
        }
        {
            let mut g = c.benchmark_group("BigStateDPF_EvalAll");
            for depth in 2..25 {
                if (1 << depth) <= 2 * points {
                    continue;
                }
                let mut ab_map = HashMap::new();
                while ab_map.len() < points {
                    let mut alpha_v = BitVec::new(depth);
                    loop {
                        alpha_v.fill_random(&mut rng);
                        if !ab_map.contains_key(&alpha_v) {
                            break;
                        }
                    }
                    let beta: Vec<bool> =
                        (0..OUTPUT_WIDTH).map(|_| rng.next_u32() & 1 == 1).collect();
                    let beta_bitvec = BitVec::from(&beta[..]);
                    ab_map.insert(alpha_v.clone(), beta_bitvec.clone());
                }
                let alphas_betas: Vec<_> = ab_map.into_iter().collect();
                g.bench_with_input(format!("{}/{}", points, depth), &depth, |b, depth| {
                    let (k_0, _) = BigStateDpfKey::gen(&alphas_betas, &roots);
                    b.iter(|| k_0.eval_all());
                });
            }
        }
    }
}
pub fn okvs_dpf_bench(c: &mut Criterion) {
    const W: usize = 5;
    for points in [5 * 5, 16 * 16, 76 * 76] {
        let mut rng = thread_rng();
        {
            let mut g = c.benchmark_group("OkvsDPF_Keygen");
            for depth in 2..25 {
                if (1 << depth) <= 2 * points {
                    continue;
                }
                let mut ab_map = HashMap::new();
                while ab_map.len() < points {
                    let mut alpha_v = BitVec::new(depth);
                    loop {
                        alpha_v.fill_random(&mut rng);
                        if !ab_map.contains_key(&alpha_v) {
                            break;
                        }
                    }
                    let beta: Vec<bool> =
                        (0..OUTPUT_WIDTH).map(|_| rng.next_u32() & 1 == 1).collect();
                    let beta_bitvec = BitVec::from(&beta[..]);
                    ab_map.insert(alpha_v.clone(), beta_bitvec.clone());
                }
                let alphas_betas: Vec<_> = ab_map.into_iter().collect();
                g.bench_with_input(format!("{}/{}", points, depth), &depth, |b, depth| {
                    let okvs_scheme = OkvsDmpf::<W, 2>::new(
                        *depth,
                        OUTPUT_WIDTH,
                        points,
                        rb_okvs::EpsilonPercent::Ten,
                    );
                    let kvs: Vec<_> = alphas_betas
                        .iter()
                        .map(|v| (v.0.as_ref()[0].into(), v.1.as_ref()[0].into()))
                        .collect();
                    b.iter(|| okvs_scheme.try_gen(&kvs, &mut rng).unwrap());
                });
            }
        }
        {
            let mut g = c.benchmark_group("OkvsDPF_Eval");
            for depth in 2..25 {
                if (1 << depth) <= 2 * points {
                    continue;
                }
                let mut ab_map = HashMap::new();
                while ab_map.len() < points {
                    let mut alpha_v = BitVec::new(depth);
                    loop {
                        alpha_v.fill_random(&mut rng);
                        if !ab_map.contains_key(&alpha_v) {
                            break;
                        }
                    }
                    let beta: Vec<bool> =
                        (0..OUTPUT_WIDTH).map(|_| rng.next_u32() & 1 == 1).collect();
                    let beta_bitvec = BitVec::from(&beta[..]);
                    ab_map.insert(alpha_v.clone(), beta_bitvec.clone());
                }
                let alphas_betas: Vec<_> = ab_map.into_iter().collect();
                g.bench_with_input(format!("{}/{}", points, depth), &depth, |b, depth| {
                    let okvs_scheme = OkvsDmpf::<W, 2>::new(
                        *depth,
                        OUTPUT_WIDTH,
                        points,
                        rb_okvs::EpsilonPercent::Ten,
                    );
                    let kvs: Vec<_> = alphas_betas
                        .iter()
                        .map(|v| (v.0.as_ref()[0].into(), v.1.as_ref()[0].into()))
                        .collect();
                    let (k_0, _) = okvs_scheme.try_gen(&kvs, &mut rng).unwrap();
                    let i = ((rng.next_u64() & ((1 << depth) - 1)) as u128).into();
                    let mut o = Default::default();
                    b.iter(|| k_0.eval(&i, &mut o));
                });
            }
        }
        {
            let mut g = c.benchmark_group("OkvsDPF_EvalAll");
            for depth in 2..25 {
                if (1 << depth) <= 2 * points {
                    continue;
                }
                let mut ab_map = HashMap::new();
                while ab_map.len() < points {
                    let mut alpha_v = BitVec::new(depth);
                    loop {
                        alpha_v.fill_random(&mut rng);
                        if !ab_map.contains_key(&alpha_v) {
                            break;
                        }
                    }
                    let beta: Vec<bool> =
                        (0..OUTPUT_WIDTH).map(|_| rng.next_u32() & 1 == 1).collect();
                    let beta_bitvec = BitVec::from(&beta[..]);
                    ab_map.insert(alpha_v.clone(), beta_bitvec.clone());
                }
                let alphas_betas: Vec<_> = ab_map.into_iter().collect();
                g.bench_with_input(format!("{}/{}", points, depth), &depth, |b, depth| {
                    let okvs_scheme = OkvsDmpf::<W, 2>::new(
                        *depth,
                        OUTPUT_WIDTH,
                        points,
                        rb_okvs::EpsilonPercent::Ten,
                    );
                    let kvs: Vec<_> = alphas_betas
                        .iter()
                        .map(|v| (v.0.as_ref()[0].into(), v.1.as_ref()[0].into()))
                        .collect();
                    let (k_0, _) = okvs_scheme.try_gen(&kvs, &mut rng).unwrap();
                    b.iter(|| k_0.eval_all());
                });
            }
        }
    }
}
pub fn batch_code_dpf_bench(c: &mut Criterion) {
    const W: usize = 4;
    const EXPANSION_OVERHEAD_IN_PERCENT: usize = 50;
    for points in [5 * 5, 16 * 16, 76 * 76] {
        let mut rng = thread_rng();
        {
            let mut g = c.benchmark_group("BatchCodeDPF_Keygen");
            for depth in 2..25 {
                if (1 << depth) <= 2 * points {
                    continue;
                }
                let mut ab_map = HashMap::new();
                while ab_map.len() < points {
                    let mut alpha_v = BitVec::new(depth);
                    loop {
                        alpha_v.fill_random(&mut rng);
                        if !ab_map.contains_key(&alpha_v) {
                            break;
                        }
                    }
                    let beta: Vec<bool> =
                        (0..OUTPUT_WIDTH).map(|_| rng.next_u32() & 1 == 1).collect();
                    let beta_bitvec = BitVec::from(&beta[..]);
                    ab_map.insert(alpha_v.clone(), beta_bitvec.clone());
                }
                let alphas_betas: Vec<_> = ab_map.into_iter().collect();
                g.bench_with_input(format!("{}/{}", points, depth), &depth, |b, depth| {
                    let scheme = BatchCodeDmpf::new(*depth, W, EXPANSION_OVERHEAD_IN_PERCENT);
                    let kvs: Vec<_> = alphas_betas
                        .iter()
                        .map(|v| (v.0.as_ref()[0].into(), v.1.as_ref()[0].into()))
                        .collect();
                    b.iter(|| scheme.try_gen(&kvs, &mut rng).unwrap());
                });
            }
        }
        {
            let mut g = c.benchmark_group("BatchCodeDPF_Eval");
            for depth in 2..25 {
                if (1 << depth) <= 2 * points {
                    continue;
                }
                let mut ab_map = HashMap::new();
                while ab_map.len() < points {
                    let mut alpha_v = BitVec::new(depth);
                    loop {
                        alpha_v.fill_random(&mut rng);
                        if !ab_map.contains_key(&alpha_v) {
                            break;
                        }
                    }
                    let beta: Vec<bool> =
                        (0..OUTPUT_WIDTH).map(|_| rng.next_u32() & 1 == 1).collect();
                    let beta_bitvec = BitVec::from(&beta[..]);
                    ab_map.insert(alpha_v.clone(), beta_bitvec.clone());
                }
                let alphas_betas: Vec<_> = ab_map.into_iter().collect();
                g.bench_with_input(format!("{}/{}", points, depth), &depth, |b, depth| {
                    let scheme = BatchCodeDmpf::new(*depth, W, EXPANSION_OVERHEAD_IN_PERCENT);
                    let kvs: Vec<_> = alphas_betas
                        .iter()
                        .map(|v| (v.0.as_ref()[0].into(), v.1.as_ref()[0].into()))
                        .collect();
                    let (k_0, _) = scheme.try_gen(&kvs, &mut rng).unwrap();
                    let i: u128 = (rng.next_u64() & ((1 << depth) - 1)) as u128;
                    let mut o = 0u128;
                    b.iter(|| k_0.eval(&i, &mut o));
                });
            }
        }
        {
            let mut g = c.benchmark_group("BatchCodeDPF_EvalAll");
            for depth in 2..25 {
                if (1 << depth) <= 2 * points {
                    continue;
                }
                let mut ab_map = HashMap::new();
                while ab_map.len() < points {
                    let mut alpha_v = BitVec::new(depth);
                    loop {
                        alpha_v.fill_random(&mut rng);
                        if !ab_map.contains_key(&alpha_v) {
                            break;
                        }
                    }
                    let beta: Vec<bool> =
                        (0..OUTPUT_WIDTH).map(|_| rng.next_u32() & 1 == 1).collect();
                    let beta_bitvec = BitVec::from(&beta[..]);
                    ab_map.insert(alpha_v.clone(), beta_bitvec.clone());
                }
                let alphas_betas: Vec<_> = ab_map.into_iter().collect();
                g.bench_with_input(format!("{}/{}", points, depth), &depth, |b, depth| {
                    let scheme = BatchCodeDmpf::new(*depth, W, EXPANSION_OVERHEAD_IN_PERCENT);
                    let kvs: Vec<_> = alphas_betas
                        .iter()
                        .map(|v| (v.0.as_ref()[0].into(), v.1.as_ref()[0].into()))
                        .collect();
                    let (k_0, _) = scheme.try_gen(&kvs, &mut rng).unwrap();
                    b.iter(|| k_0.eval_all());
                });
            }
        }
    }
}
criterion_group!(
    benches,
    dpf_bench,
    bigstate_dpf_bench,
    okvs_dpf_bench,
    batch_code_dpf_bench
);
criterion_main!(benches);
