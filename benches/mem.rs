use criterion::{criterion_group, criterion_main, Criterion};
use dmpf::Node;
use rand::{thread_rng, RngCore};
fn bench_mem(c: &mut Criterion) {
    const MAX_LOG_SIZE: usize = 20;
    const MIN_LOG_SIZE: usize = 6;
    let mut rng = thread_rng();
    let mut g = c.benchmark_group("mem_random");
    for log_size in (MIN_LOG_SIZE..=MAX_LOG_SIZE).step_by(2) {
        let mut v: Vec<_> = (0u64..1 << log_size).collect();
        for i in (0..v.len()).rev() {
            let idx = (rng.next_u64() as usize) % (i + 1);
            v.swap(idx, i);
        }
        g.throughput(criterion::Throughput::Elements(()))
    }
}
criterion_group!(benches, bench_mem);
criterion_main!(benches);
