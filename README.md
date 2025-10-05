# Fast Monte Carlo π Estimator (Rust + AVX2)

A high-performance Monte Carlo simulation written in **Rust**, specialized for estimating the value of π.  
This implementation is **CPU-only**, optimized with **AVX2 SIMD intrinsics** via [`std::arch::x86_64`].  
**Note:** This project is limited to **x86_64 CPUs with AVX2 support**. No fallback implementation is provided.

## Features
- Optimized using **AVX2 vectorization** for maximum throughput
- Multi-threaded execution across CPU cores
- Designed specifically for **π estimation** via Monte Carlo sampling

## Requirements
- Rust 1.87.0
- **x86_64 architecture only** (uses `std::arch::x86_64`)  
- CPU with **AVX2 support**   
- tested on Windows  

## Installation
```bash
git clone https://github.com/yua134/montecarlo-pi.git
cd montecarlo-pi
cargo build --release
```

Example output:
```
interior:204203538220 total:260000000000 pi:3.1415928956923076 time: 14.5927911s
```
Tested on: Windows 11, Intel Core ultra5-125U

## Changing the number of trials
The number of trials is defined as a constant in the source code.  
If you want to change it, edit the value of the constant in `src/main.rs`:

```rust
const MAX: usize = 150_000_000;
```

For example, to run with 10 million trials:

```rust
const MAX: usize = 10_000_000;
```

Then rebuild with:

```bash
cargo build --release
```

## License
MIT License
