// fn bench_ole_pcg_regular(c: &mut Criterion)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dmpf::{
    batch_code::BatchCodeDmpf, big_state::BigStateDmpf, okvs::OkvsDmpf, Dmpf, DpfDmpf,
    PrimeField64, PrimeField64x2, RadixTwoFftFriendFieldElement, SmallFieldContainer,
};
use ole_pcg::{
    gen, gen_regular,
    ring::{ModuloPolynomial, TwoPowerDegreeCyclotomicPolynomial},
    OlePcgSeed,
};

fn bench_ole_pcg_regular_single<
    const W: usize,
    C: SmallFieldContainer<W, F>,
    D: Dmpf<C>,
    F: RadixTwoFftFriendFieldElement,
>(
    c: &mut Criterion,
    d: &D,
    dmpf_title: &str,
    log_degree: usize,
    compression_factor: usize,
    weight: usize,
    modulo: impl ModuloPolynomial<F>,
) {
    c.bench_with_input(
        BenchmarkId::new(
            "gen/regular",
            format!(
                "dmpf:{}/c:{}/t:{}/logN:{}",
                dmpf_title, compression_factor, weight, log_degree
            ),
        ),
        &(log_degree, compression_factor, weight, d),
        |b, input| {
            let (log_degree, compression_factor, weight, d) = *input;
            b.iter(|| gen_regular(log_degree, compression_factor, modulo.clone(), weight, d))
        },
    );
    let (k_0, k_1) = gen_regular(log_degree, compression_factor, modulo, weight, d);
    c.bench_with_input(
        BenchmarkId::new(
            "expand/regular",
            format!(
                "dmpf:{}/c:{}/t:{}/logN:{}",
                dmpf_title, compression_factor, weight, log_degree
            ),
        ),
        &(log_degree, compression_factor, weight, d),
        |b, input| {
            let (log_degree, compression_factor, weight, d) = *input;
            b.iter(|| k_0.expand())
        },
    );
}
fn bench_ole_pcg_nonregular_single<
    const W: usize,
    C: SmallFieldContainer<W, F>,
    D: Dmpf<C>,
    F: RadixTwoFftFriendFieldElement,
>(
    c: &mut Criterion,
    d: &D,
    dmpf_title: &str,
    log_degree: usize,
    compression_factor: usize,
    weight: usize,
    modulo: impl ModuloPolynomial<F>,
) {
    c.bench_with_input(
        BenchmarkId::new(
            "gen/nonregular",
            format!(
                "dmpf:{}/c:{}/t:{}/logN:{}",
                dmpf_title, compression_factor, weight, log_degree
            ),
        ),
        &(log_degree, compression_factor, weight, d),
        |b, input| {
            let (log_degree, compression_factor, weight, d) = *input;
            b.iter(|| gen(log_degree, compression_factor, modulo.clone(), weight, d))
        },
    );
    let (k_0, k_1) = gen_regular(log_degree, compression_factor, modulo, weight, d);
    c.bench_with_input(
        BenchmarkId::new(
            "expand/nonregular",
            format!(
                "dmpf:{}/c:{}/t:{}/logN:{}",
                dmpf_title, compression_factor, weight, log_degree
            ),
        ),
        &(log_degree, compression_factor, weight, d),
        |b, input| {
            let (log_degree, compression_factor, weight, d) = *input;
            b.iter(|| k_0.expand())
        },
    );
}
fn bench_ole_pcg_with_dmpf<
    const W: usize,
    F: RadixTwoFftFriendFieldElement,
    C: SmallFieldContainer<W, F>,
    D: Dmpf<C>,
>(
    d: &D,
    dmpf_title: &str,
    c: &mut Criterion,
) {
    const LOG_DEG: usize = 20;
    const PARAMS: [(usize, usize); 3] = [(66, 2), (14, 4), (8, 5)];

    let modulo = TwoPowerDegreeCyclotomicPolynomial::<F>::new(LOG_DEG);
    for (weight, compression_factor) in PARAMS {
        bench_ole_pcg_regular_single(
            c,
            d,
            dmpf_title,
            LOG_DEG,
            compression_factor,
            weight,
            modulo.clone(),
        );
        bench_ole_pcg_nonregular_single(
            c,
            d,
            dmpf_title,
            LOG_DEG,
            compression_factor,
            weight,
            modulo.clone(),
        );
    }
}
fn bench_ole_pcg(c: &mut Criterion) {
    let dpf = DpfDmpf::new();
    bench_ole_pcg_with_dmpf::<2, PrimeField64, PrimeField64x2, _>(&dpf, "Dpf", c);
    const W: usize = 200;
    const BIN_W: usize = W.div_ceil(64);
    let dpf = OkvsDmpf::<BIN_W, W, PrimeField64x2>::new(dmpf::EpsilonPercent::Ten);
    bench_ole_pcg_with_dmpf::<2, PrimeField64, PrimeField64x2, _>(&dpf, "Okvs", c);
    let dpf = BigStateDmpf::new();
    bench_ole_pcg_with_dmpf::<2, PrimeField64, PrimeField64x2, _>(&dpf, "BigState", c);
    let dpf = BatchCodeDmpf::new();
    bench_ole_pcg_with_dmpf::<2, PrimeField64, PrimeField64x2, _>(&dpf, "BatchCode", c);
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10).configure_from_args();
    targets = bench_ole_pcg);
criterion_main!(benches);
