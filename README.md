# Improved Constructions for Distributed Multi-Point Functions

This repository contains the code for the the "Improved Constructions for Distributed Multi-Point Functions" by Elette Boyle, Niv Gilboa, Matan Hamillis, Yuval Ishai and Yaxin Tu, accepted to IEEE S&P 2025.

For further information on the benchmarks please use the `benches` directory.

## Useful FFT Benchmarks
We provide some useful FFT benchmarks in the `ole-pcg/benches/fft.rs` file. 
To run them for benchmark sizes of $$2^{17}$$ to $$2^{25}$$, use the following command:

```bash
FFT_BENCH_MIN_LOG_SIZE=17 FFT_BENCH_MAX_LOG_SIZE=25 cargo bench --workspace --bench fft 
```

The FFT benchmarks are over a 64-bit prime field, and are single-threaded.
