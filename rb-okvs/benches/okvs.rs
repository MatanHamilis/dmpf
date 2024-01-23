use criterion::{criterion_group, criterion_main, Criterion};
use rand::thread_rng;
use rb_okvs::{g, EpsilonPercent, LogN, OkvsValueU128Array};

const LAMBDA: usize = 40;
fn generate_kvs<const WIDTH: usize>(
    size: usize,
) -> Vec<(OkvsValueU128Array<WIDTH>, OkvsValueU128Array<WIDTH>)> {
    let mut rng = thread_rng();
    (0..size).map(|i| (generate_kv(i as u128))).collect()
}
fn generate_kv<const WIDTH: usize>(
    i: u128,
) -> (OkvsValueU128Array<WIDTH>, OkvsValueU128Array<WIDTH>) {
    ([i; WIDTH].into(), [i; WIDTH].into())
}
fn bench_encode(c: &mut Criterion) {
    const WIDTH: usize = 2;
    let kvs = generate_kvs(1 << 24);
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
                            rb_okvs::encode::<3, _, OkvsValueU128Array<WIDTH>>(
                                &kvs[..1 << input_log_usize],
                                epsilon_percent,
                            )
                        });
                    }
                    4 => {
                        b.iter(|| {
                            rb_okvs::encode::<4, _, OkvsValueU128Array<WIDTH>>(
                                &kvs[..1 << input_log_usize],
                                epsilon_percent,
                            )
                        });
                    }
                    5 => {
                        b.iter(|| {
                            rb_okvs::encode::<5, _, OkvsValueU128Array<WIDTH>>(
                                &kvs[..1 << input_log_usize],
                                epsilon_percent,
                            )
                        });
                    }
                    6 => {
                        b.iter(|| {
                            rb_okvs::encode::<6, _, OkvsValueU128Array<WIDTH>>(
                                &kvs[..1 << input_log_usize],
                                epsilon_percent,
                            )
                        });
                    }
                    7 => {
                        b.iter(|| {
                            rb_okvs::encode::<7, _, OkvsValueU128Array<WIDTH>>(
                                &kvs[..1 << input_log_usize],
                                epsilon_percent,
                            )
                        });
                    }
                    8 => {
                        b.iter(|| {
                            rb_okvs::encode::<8, _, OkvsValueU128Array<WIDTH>>(
                                &kvs[..1 << input_log_usize],
                                epsilon_percent,
                            )
                        });
                    }
                    9 => {
                        b.iter(|| {
                            rb_okvs::encode::<9, _, OkvsValueU128Array<WIDTH>>(
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

fn bench_decode(c: &mut Criterion) {
    const WIDTH: usize = 2;;
    let kvs = generate_kvs(1 << 24);
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
            group.bench_function(format!("decode/log_n:{}", usize::from(input_log)), |b| {
                let w = g(LAMBDA, epsilon_percent, input_log);
                // We know W will range between 4 and 9.
                match w {
                    3 => {
                        let c = rb_okvs::encode::<3, _, OkvsValueU128Array<2>>(
                            &kvs[..1 << input_log_usize],
                            epsilon_percent,
                        );
                        b.iter(|| c.decode(first_key));
                    }
                    4 => {
                        let c = rb_okvs::encode::<4, _, OkvsValueU128Array<WIDTH>>(
                            &kvs[..1 << input_log_usize],
                            epsilon_percent,
                        );
                        b.iter(|| c.decode(first_key));
                    }
                    5 => {
                        let c = rb_okvs::encode::<5, _, OkvsValueU128Array<WIDTH>>(
                            &kvs[..1 << input_log_usize],
                            epsilon_percent,
                        );
                        b.iter(|| c.decode(first_key));
                    }
                    6 => {
                        let c = rb_okvs::encode::<6, _, OkvsValueU128Array<WIDTH>>(
                            &kvs[..1 << input_log_usize],
                            epsilon_percent,
                        );
                        b.iter(|| c.decode(first_key));
                    }
                    7 => {
                        let c = rb_okvs::encode::<7, _, OkvsValueU128Array<WIDTH>>(
                            &kvs[..1 << input_log_usize],
                            epsilon_percent,
                        );
                        b.iter(|| c.decode(first_key));
                    }
                    8 => {
                        let c = rb_okvs::encode::<8, _, OkvsValueU128Array<WIDTH>>(
                            &kvs[..1 << input_log_usize],
                            epsilon_percent,
                        );
                        b.iter(|| c.decode(first_key));
                    }
                    9 => {
                        let c = rb_okvs::encode::<9, _, OkvsValueU128Array<WIDTH>>(
                            &kvs[..1 << input_log_usize],
                            epsilon_percent,
                        );
                        b.iter(|| c.decode(first_key));
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

criterion_group!(benches, bench_decode, bench_encode);
criterion_main!(benches);
