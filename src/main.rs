use std::{arch::x86_64::*, sync::atomic::AtomicUsize, time::Instant};

use rayon::iter::{IntoParallelIterator, ParallelIterator};

const MAX: usize = 2_500_000_000;
const CORES: usize = 13;

fn main() {
    let start = Instant::now();
    let (pi, count, total) = pi();
    let duration = Instant::now() - start;
    println!(
        "interior:{:?} total:{:?} pi:{:?} time: {:?}",
        count, total, pi, duration
    );
}

#[inline(always)]
fn pi() -> (f64, usize, usize) {
    let count = AtomicUsize::new(0);
    (0..CORES).into_par_iter().for_each(|_| unsafe {
        let seed = rand::random::<u64>();
        let mut rng = Xoroshiro128PlusPlusSimd::from_seed(seed);
        let mut c = 0;
        for _ in 0..MAX {
            let v = rng.next_u64x4();
            let mantissa = _mm256_srli_epi32(v, 9);
            let exp = _mm256_set1_epi32(127 << 23);
            let bits = _mm256_or_si256(exp, mantissa);
            let floats = _mm256_castsi256_ps(bits);
            let one = _mm256_set1_ps(1.0);
            let xy = _mm256_sub_ps(floats, one);
            let x = _mm256_mul_ps(xy, xy);
            let y = _mm256_shuffle_ps(x, x, 0xB1);
            let sum = _mm256_add_ps(x, y);
            let sub = _mm256_sub_ps(sum, one);
            let mask = _mm256_movemask_ps(sub);
            c += _popcnt32(mask) as usize;
        }
        count.fetch_add(c, std::sync::atomic::Ordering::Release);
    });
    let counts = count.load(std::sync::atomic::Ordering::Acquire);
    let total = MAX * CORES * 8;
    (counts as f64 * 4.0 / total as f64, counts, total)
}

#[inline(always)]
unsafe fn rotl_epi64<const K: i32, const K1: i32>(x: __m256i) -> __m256i {
    unsafe {
        let left = _mm256_slli_epi64(x, K);
        let right = _mm256_srli_epi64(x, K1);
        _mm256_or_si256(left, right)
    }
}

#[derive(Clone, Copy)]
struct Xoroshiro128PlusPlusSimd {
    s0: __m256i,
    s1: __m256i,
}

impl Xoroshiro128PlusPlusSimd {
    #[inline(always)]
    unsafe fn from_seed(seed: u64) -> Self {
        unsafe {
            let mut seeds0 = [0u64; 4];
            let mut seeds1 = [0u64; 4];
            for i in 0..4 {
                seeds0[i] = seed.wrapping_add(i as u64 * 0x9E3779B97F4A7C15);
                seeds1[i] = seed.wrapping_add((i as u64 + 4) * 0x9E3779B97F4A7C15);
            }
            let s0 = _mm256_set_epi64x(
                seeds0[3] as i64,
                seeds0[2] as i64,
                seeds0[1] as i64,
                seeds0[0] as i64,
            );
            let s1 = _mm256_set_epi64x(
                seeds1[3] as i64,
                seeds1[2] as i64,
                seeds1[1] as i64,
                seeds1[0] as i64,
            );
            Self { s0, s1 }
        }
    }

    #[inline(always)]
    unsafe fn next_u64x4(&mut self) -> __m256i {
        unsafe {
            let sum = _mm256_add_epi64(self.s0, self.s1);
            let rot = rotl_epi64::<17, 47>(sum);
            let result = _mm256_add_epi64(rot, self.s0);

            let t = _mm256_slli_epi64(self.s1, 9);
            let mut s1 = _mm256_xor_si256(self.s1, self.s0);
            let mut s0 = rotl_epi64::<49, 15>(self.s0);
            s0 = _mm256_xor_si256(s0, s1);
            s0 = _mm256_xor_si256(s0, t);
            s1 = rotl_epi64::<28, 36>(s1);

            self.s0 = s0;
            self.s1 = s1;

            result
        }
    }
}
