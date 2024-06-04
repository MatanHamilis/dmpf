use aes::Aes128;
use criterion::{criterion_group, criterion_main, Criterion};
use fpe::ff1::BinaryNumeralString;

fn bench_fpe(c: &mut Criterion) {
    c.bench_function("fpe", |b| {
        let k = vec![0; 16];
        let f = fpe::ff1::FF1::<Aes128>::new(&k, 2).unwrap();
        let v = [1; 16];
        let s = BinaryNumeralString::from_bytes_le(&v[..]);
        b.iter(|| {
            f.encrypt(&[], &s).unwrap();
        });
    });
}

criterion_group!(bench_group, bench_fpe);
criterion_main!(bench_group);
