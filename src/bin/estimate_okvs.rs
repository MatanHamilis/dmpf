use std::collections::HashMap;

use aes_prng::AesRng;
use dmpf::{
    rb_okvs::{self, OkvsBool, OkvsValue},
    DpfOutput, Node,
};
use rand::RngCore;

fn main() {
    const N: usize = 1 << 10;
    const TESTS: usize = 1 << 15;
    let mut failures = 0;
    let mut rng = AesRng::from_random_seed();
    let kvs: Vec<_> = (0..N)
        .map(|_| (Node::random(&mut rng), OkvsBool::random(&mut rng)))
        .collect();
    for i in 0..TESTS {
        if i % 100 == 0 {
            println!("Test: {i}, failures: {failures}");
        }
        let mut seed = [0u8; 16];
        rng.fill_bytes(&mut seed);
        let okvs = rb_okvs::try_encode::<20, _, _>(&kvs, dmpf::EpsilonPercent::Hundred, 1, seed);
        if okvs.is_none() {
            failures += 1;
        }
    }
    println!("Test: {TESTS}, failures: {failures}");
}
