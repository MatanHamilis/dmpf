use core::panic;

use criterion::{criterion_group, criterion_main, Criterion};
use dmpf::rb_okvs::{g, EpsilonPercent, LogN, OkvsKey, OkvsU128, OkvsValue};
use dmpf::PrimeField64x2;
use rand::{thread_rng, CryptoRng, RngCore};

const LAMBDA: usize = 40;
fn generate_kvs<K: OkvsKey, V: OkvsValue>(size: usize) -> Vec<(K, V)> {
    let mut rng = thread_rng();
    (0..size).map(|_| (generate_kv(&mut rng))).collect()
}
fn generate_kv<K: OkvsKey, V: OkvsValue, R: RngCore + CryptoRng>(mut rng: R) -> (K, V) {
    (K::random(&mut rng), V::random(&mut rng))
}
fn do_bench_encode<K: OkvsKey, V: OkvsValue>(c: &mut Criterion) {
    let kvs = generate_kvs::<K, V>(1 << 24);
    for epsilon_percent in [
        EpsilonPercent::Three,
        EpsilonPercent::Five,
        EpsilonPercent::Seven,
        EpsilonPercent::Ten,
    ] {
        let mut group = c.benchmark_group(format!("epsilon:{}", usize::from(epsilon_percent)));
        for input_log in [
            LogN::Ten,
            LogN::Fourteen,
            LogN::Sixteen,
            LogN::Eighteen,
            LogN::Twenty,
            LogN::TwentyFour,
        ] {
            let input_log_usize = usize::from(input_log);
            group.bench_function(format!("encode/log_n:{}", usize::from(input_log)), |b| {
                let w = g(LAMBDA, epsilon_percent, input_log);
                // We know W will range between 4 and 9.
                match w {
                    3 => {
                        b.iter(|| {
                            rb_okvs::encode::<3, _, _>(
                                &kvs[..1 << input_log_usize],
                                epsilon_percent,
                            )
                        });
                    }
                    4 => {
                        b.iter(|| {
                            rb_okvs::encode::<4, _, _>(
                                &kvs[..1 << input_log_usize],
                                epsilon_percent,
                            )
                        });
                    }
                    5 => {
                        b.iter(|| {
                            rb_okvs::encode::<5, _, _>(
                                &kvs[..1 << input_log_usize],
                                epsilon_percent,
                            )
                        });
                    }
                    6 => {
                        b.iter(|| {
                            rb_okvs::encode::<6, _, _>(
                                &kvs[..1 << input_log_usize],
                                epsilon_percent,
                            )
                        });
                    }
                    7 => {
                        b.iter(|| {
                            rb_okvs::encode::<7, _, _>(
                                &kvs[..1 << input_log_usize],
                                epsilon_percent,
                            )
                        });
                    }
                    8 => {
                        b.iter(|| {
                            rb_okvs::encode::<8, _, _>(
                                &kvs[..1 << input_log_usize],
                                epsilon_percent,
                            )
                        });
                    }
                    9 => {
                        b.iter(|| {
                            rb_okvs::encode::<9, _, _>(
                                &kvs[..1 << input_log_usize],
                                epsilon_percent,
                            )
                        });
                    }
                    _ => panic!(
                        "w={}, LAMBDA={}, epsilon_percent={}, log_n={}",
                        w,
                        LAMBDA,
                        usize::from(epsilon_percent),
                        usize::from(input_log)
                    ),
                };
            });
        }
    }
}

fn do_bench_decode<K: OkvsKey, V: OkvsValue>(c: &mut Criterion) {
    let kvs = generate_kvs::<K, V>(1 << 24);
    let first_key = &kvs[0].0;
    for epsilon_percent in [
        EpsilonPercent::Three,
        EpsilonPercent::Five,
        EpsilonPercent::Seven,
        EpsilonPercent::Ten,
    ] {
        let mut group = c.benchmark_group(format!("epsilon:{}", usize::from(epsilon_percent)));
        for input_log in [
            LogN::Ten,
            LogN::Fourteen,
            LogN::Sixteen,
            LogN::Eighteen,
            LogN::Twenty,
            LogN::TwentyFour,
        ] {
            let input_log_usize = usize::from(input_log);
            let w = g(LAMBDA, epsilon_percent, input_log);
            // We know W will range between 4 and 9.
            match w {
                540 => {
                    let c =
                        rb_okvs::encode::<540, _, _>(&kvs[..1 << input_log_usize], epsilon_percent);
                    group.bench_function(format!("decode/log_n:{}", usize::from(input_log)), |b| {
                        b.iter(|| c.decode(first_key));
                    });
                }
                554 => {
                    let c =
                        rb_okvs::encode::<554, _, _>(&kvs[..1 << input_log_usize], epsilon_percent);
                    group.bench_function(format!("decode/log_n:{}", usize::from(input_log)), |b| {
                        b.iter(|| c.decode(first_key));
                    });
                }
                570 => {
                    let c =
                        rb_okvs::encode::<570, _, _>(&kvs[..1 << input_log_usize], epsilon_percent);
                    group.bench_function(format!("decode/log_n:{}", usize::from(input_log)), |b| {
                        b.iter(|| c.decode(first_key));
                    });
                }
                6 => {
                    let c =
                        rb_okvs::encode::<6, _, _>(&kvs[..1 << input_log_usize], epsilon_percent);
                    group.bench_function(format!("decode/log_n:{}", usize::from(input_log)), |b| {
                        b.iter(|| c.decode(first_key));
                    });
                }
                7 => {
                    let c =
                        rb_okvs::encode::<7, _, _>(&kvs[..1 << input_log_usize], epsilon_percent);
                    group.bench_function(format!("decode/log_n:{}", usize::from(input_log)), |b| {
                        b.iter(|| c.decode(first_key));
                    });
                }
                8 => {
                    let c =
                        rb_okvs::encode::<8, _, _>(&kvs[..1 << input_log_usize], epsilon_percent);
                    group.bench_function(format!("decode/log_n:{}", usize::from(input_log)), |b| {
                        b.iter(|| c.decode(first_key));
                    });
                }
                9 => {
                    let c =
                        rb_okvs::encode::<9, _, _>(&kvs[..1 << input_log_usize], epsilon_percent);
                    group.bench_function(format!("decode/log_n:{}", usize::from(input_log)), |b| {
                        b.iter(|| c.decode(first_key));
                    });
                }
                _ => panic!(
                    "w={}, LAMBDA={}, epsilon_percent={}, log_n={}",
                    w,
                    LAMBDA,
                    usize::from(epsilon_percent),
                    usize::from(input_log)
                ),
            };
        }
    }
}
fn bench_encode(c: &mut Criterion) {
    do_bench_encode::<OkvsU128, PrimeField64x2>(c);
}
fn bench_decode(c: &mut Criterion) {
    do_bench_decode::<OkvsU128, PrimeField64x2>(c);
}

criterion_group!(benches, bench_decode, bench_encode);
criterion_main!(benches);
