use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dmpf::{prg::double_prg_many, Node};
use rand::thread_rng;
fn bench_prg(c: &mut Criterion) {
    const MIN_LOG_SIZE: usize = 4;
    const MAX_LOG_SIZE: usize = 20;
    let mut group = c.benchmark_group("many_prg");
    let mut rng = thread_rng();
    let v_in: Vec<_> = (0..1 << MAX_LOG_SIZE)
        .map(|_| Node::random(&mut rng))
        .collect();
    let mut v: Vec<_> = (0..2 << MAX_LOG_SIZE)
        .map(|_| Node::random(&mut rng))
        .collect();
    const CHILDREN: [u8; 2] = [0, 1];
    for log_size in MIN_LOG_SIZE..=MAX_LOG_SIZE {
        group.throughput(criterion::Throughput::Elements(1 << log_size));
        group.bench_with_input(
            BenchmarkId::from_parameter(log_size),
            &log_size,
            |b, &log_size| {
                b.iter(|| {
                    double_prg_many(&v_in[..1 << log_size], &CHILDREN, &mut v[..2 << log_size])
                })
            },
        );
    }
}
criterion_group!(benches, bench_prg);
criterion_main!(benches);
