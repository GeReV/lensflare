use crate::fft::complex::Complex;
use std::f32::consts::TAU;
use rayon::prelude::*;

pub mod complex;

fn calculate_twiddles(n: usize) -> Box<[Complex]> {
    let mut result = Vec::with_capacity(n);

    for k in 0..n {
        result.push(Complex::exp(-TAU * k as f32 / n as f32));
    }

    result.into_boxed_slice()
}

fn fft_step_stockham(
    data: &[Complex],
    twiddles: &[Complex],
    x: usize,
    n: usize,
    n_s: usize,
) -> Complex {
    let base = x / n_s * (n_s / 2);
    let offset = x % (n_s / 2);

    let idx0 = base + offset;
    let idx1 = idx0 + n / 2;

    let a = data[idx0];
    let b = data[idx1];

    let e_idx = x * (n / n_s) % n;
    let e = twiddles[e_idx];

    a + b * e
}

pub fn fft_stockham(data: &mut [Complex]) {
    let len = data.len();

    assert!(len.is_power_of_two());

    let twiddles = calculate_twiddles(len);

    // TODO: Any way to avoid this and the later copy?
    let mut a = Vec::from(&*data);
    let mut b = vec![Complex::ZERO; len];

    for iter in 0..len.ilog2() {
        let n_s = 1 << (iter + 1);

        for x in 0..len {
            b[x] = fft_step_stockham(&a, &twiddles, x, len, n_s);
        }

        std::mem::swap(&mut a, &mut b);
    }

    data.copy_from_slice(&a);
}

pub fn fftshift<T: Send>(data: &mut [T], width: usize) {
    // Swap bottom rows with top rows.
    let (a, b) = data.split_at_mut(data.len() / 2);
    a.swap_with_slice(b);

    data.par_chunks_exact_mut(width).for_each(|row_data| {
        let (a, b) = row_data.split_at_mut(width / 2);

        a.swap_with_slice(b);
    });
}


// fn fft_real<const N: usize>(data: &[f32; N]) -> Box<[f32; N]> {
//     debug_assert!(N.is_power_of_two());
//
//     let twiddles = calculate_twiddles(N);
//
//     let (mut a, mut b) = (Box::new([Complex::ZERO; N]), Box::new([Complex::ZERO; N]));
//
//     for (a, &x) in a.iter_mut().zip(data.iter()) {
//         a.re = x;
//     }
//
//     for iter in 0..N.ilog2() {
//         let n_s = 1 << (iter + 1);
//         let half = n_s / 2;
//         let num_blocks = N / n_s;
//
//         for block in 0..num_blocks {
//             let out_base = block * half;
//
//             // only compute the first half of the output butterflies
//             for k in 0..half {
//                 let x = out_base + k; // same x indexing your Stockham mapper expects
//                 b[x] = fft_step_stockham(&*a, &twiddles, x, N, n_s);
//             }
//         }
//
//         std::mem::swap(&mut a, &mut b);
//     }
//
//     a.map(|x| x.re).into()
// }

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rustfft::{num_complex::Complex32, FftPlanner};

    // This is pretty lax, but it's good enough for now.
    const EPSILON: f32 = 1e-3;

    fn rustfft_reference(input: &[Complex32]) -> Vec<Complex32> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(input.len());
        let mut buffer = input.to_vec();
        fft.process(&mut buffer);
        buffer
    }

    fn approx_eq(a: &[Complex], b: &[Complex32], eps: f32) {
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x.re - y.re).abs() < eps,
                "Result {i} real values differ: {:?} vs {:?}",
                x.re,
                y.re
            );
            assert!(
                (x.im - y.im).abs() < eps,
                "Result {i} imaginary values differ: {:?} vs {:?}",
                x.im(),
                y.im
            );
        }
    }

    fn run_case(input: Box<[Complex]>, eps: f32) {
        let mut mine = input.clone();
        fft_stockham(&mut mine);

        let reference = rustfft_reference(&input.iter().map(|x| Complex32::new(x.re, x.im)).collect::<Vec<_>>());

        approx_eq(&mine, &reference, eps);
    }

    #[test]
    fn test_constant_one() {
        let data = [Complex::new(1.0, 0.0); 8];

        run_case(data.into(), EPSILON);
    }

    // #[test]
    // fn test_constant_one_real() {
    //     const EPSILON: f32 = 1e-5;
    //
    //     let mut proper = [Complex32::new(1.0, 0.0); 8];
    //
    //     let mut planner = FftPlanner::new();
    //     let f = planner.plan_fft_forward(8);
    //
    //     f.process(&mut proper);
    //
    //     let data = [1.0; 8];
    //
    //     let a = fft_real(&data);
    //
    //     for (i, (a, b)) in a.iter().zip(proper.iter()).enumerate() {
    //         assert!((a - b.re).abs() < EPSILON, "Result {i}, real values differ: {:?} vs {:?}", a, b.re);
    //     }
    // }

    #[test]
    fn test_random_sizes() {
        let mut rng = rand::rng();

        let mut rand_complex = || Complex::new(rng.random(), rng.random());

        // 8
        {
            let mut arr = [Complex::ZERO; 8];

            arr.fill_with(&mut rand_complex);

            run_case(arr.into(), EPSILON);
        }

        // 16
        {
            let mut arr = [Complex::ZERO; 16];

            arr.fill_with(&mut rand_complex);

            run_case(arr.into(), EPSILON);
        }
    }

    #[test]
    fn test_impulse() {
        let mut arr = [Complex::ZERO; 8];

        arr[0] = Complex::new(1.0, 0.0);

        run_case(arr.into(), EPSILON);
    }

    #[test]
    fn test_constant() {
        let arr = [Complex::new(2.0, 1.0); 16];

        run_case(arr.into(), EPSILON);
    }

    #[test]
    fn test_single_tone_sine() {
        const N: usize = 64;

        let freq = 5.0; // cycles over N samples

        let mut input = [Complex::ZERO; N];

        for i in 0..N {
            let v = (TAU * freq * i as f32 / N as f32).sin();
            input[i] = Complex::new(v, 0.0);
        }

        run_case(input.into(), EPSILON);
    }

    #[test]
    fn test_single_tone_cosine() {
        const N: usize = 64;
        let freq = 7.0;

        let mut input = [Complex::ZERO; N];
        for i in 0..N {
            let v = (TAU * freq * i as f32 / N as f32).cos();
            input[i] = Complex::new(v, 0.0);
        }

        run_case(input.into(), EPSILON);
    }

    #[test]
    fn test_two_tone_signal() {
        const N: usize = 128;
        let f1 = 3.0;
        let f2 = 17.0;

        let mut input = [Complex::ZERO; N];

        for i in 0..N {
            let v = (TAU * f1 * i as f32 / N as f32).sin()
                + 0.5 * (TAU * f2 * i as f32 / N as f32).sin();
            input[i] = Complex::new(v, 0.0);
        }

        run_case(input.into(), EPSILON);
    }

    #[test]
    fn test_chirp_signal() {
        const N: usize = 128;

        let mut input = [Complex::ZERO; N];
        for i in 0..N {
            let f = 1.0 + 20.0 * (i as f32 / N as f32); // sweep from 1→21 cycles
            let v = (TAU * f * (i as f32 / N as f32)).sin();

            input[i] = Complex::new(v, 0.0);
        }

        run_case(input.into(), EPSILON);
    }

    #[test]
    fn test_even_symmetric_real() {
        const N: usize = 32;

        let mut input = [Complex::ZERO; N];

        for i in 0..N / 2 {
            let v = (i as f32).powi(2);
            input[i] = Complex::new(v, 0.0);
            input[N - i - 1] = Complex::new(v, 0.0);
        }

        run_case(input.into(), EPSILON);
    }

    #[test]
    fn test_odd_symmetric_real() {
        const N: usize = 32;

        let mut input = [Complex::ZERO; N];

        for i in 0..N / 2 {
            let v = (i as f32).sin();

            input[i] = Complex::new(v, 0.0);
            input[N - i - 1] = Complex::new(-v, 0.0);
        }

        run_case(input.into(), EPSILON);
    }

    #[test]
    fn test_offset_impulse() {
        const N: usize = 64;

        let mut input = [Complex::ZERO; N];

        input[7] = Complex::new(1.0, 0.0);

        run_case(input.into(), EPSILON);
    }
}
