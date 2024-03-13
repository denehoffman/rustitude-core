use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustitude::prelude::*;

fn add_benchmark(c: &mut Criterion) {
    let momentum1 = FourMomentum::new(1.0, 2.0, 3.0, 4.0);
    let momentum2 = FourMomentum::new(5.0, 6.0, 7.0, 8.0);

    c.bench_function("Addition", |b| {
        b.iter(|| {
            let result = momentum1 + momentum2;
            black_box(result);
        })
    });
}

criterion_group!(benches, add_benchmark,);
criterion_main!(benches);
