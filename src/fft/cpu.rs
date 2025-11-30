use std::f32::consts::TAU;
use num_complex::Complex32;
use rayon::prelude::*;
use crate::fft::utils;

pub(crate) fn fftshift<T: Send>(data: &mut [T], width: usize) {
    debug_assert_eq!(data.len(), width * width);

    data.rotate_right(data.len() / 2);

    // Swap each row's halves.
    data.par_chunks_exact_mut(width).for_each(|row_data| {
        row_data.rotate_left(width / 2);
    });
}

fn fft_step_stockham(data: &[Complex32], x: usize, n: usize, n_s: usize) -> Complex32 {
    let base = x / n_s * (n_s / 2);
    let offset = x % (n_s / 2);

    let idx0 = base + offset;
    let idx1 = idx0 + n / 2;

    let a = data[idx0];
    let b = data[idx1];

    let e = Complex32::cis(-TAU * x as f32 / n_s as f32);

    a + b * e
}

pub fn fft_stockham(data: &mut [Complex32]) {
    let len = data.len();

    assert!(len.is_power_of_two());

    // TODO: Any way to avoid this and the later copy?
    let mut a = Vec::from(&*data);
    let mut b = vec![Complex32::ZERO; len];

    for iter in 0..len.ilog2() {
        let n_s = 1 << (iter + 1);

        for x in 0..len {
            b[x] = fft_step_stockham(&a, x, len, n_s);
        }

        std::mem::swap(&mut a, &mut b);
    }

    data.copy_from_slice(&a);
}

pub fn fft_stockham_naive(data: &mut [Complex32]) {
    let len = data.len();

    assert!(len.is_power_of_two());

    let mut a = Vec::from(&*data);

    fn fft(v: &mut [Complex32], off: usize, n: usize) {
        if n <= 1 {
            return;
        }

        let half = n / 2;
        let mid = off + half;

        for i in 0..half {
            let x = v[off + i];
            let y = v[mid + i];

            v[off + i] = x + y;
            v[mid + i] = (x - y) * Complex32::cis(-TAU * i as f32 / n as f32);
        }

        fft(v, off, half);
        fft(v, mid, half);
    }

    fft(&mut a, 0, len);

    data.copy_from_slice(&bit_reverse_permutation(&a));
}

fn bit_reverse_permutation(data: &[Complex32]) -> Vec<Complex32> {
    let len = data.len();
    assert!(len.is_power_of_two());

    let s = 1 + data.len().leading_zeros();

    (0..len).map(|i| data[i.reverse_bits() >> s]).collect()
}

fn fft_rows(data: &mut [Complex32], width: usize) {
    data.par_chunks_mut(width).for_each(fft_stockham);
}

pub(crate) fn fft2d(data: &mut [Complex32], size: usize) {
    assert!(size.is_power_of_two());

    fft_rows(data, size);

    const N: usize = 32;
    utils::transpose_chunk::<_, N>(data, size);
    utils::transpose_blocks::<_, N>(data, size);

    fft_rows(data, size);

    fftshift(data, size);
}

pub(crate) fn ifft2d(data: &mut [Complex32], size: usize) {
    assert!(size.is_power_of_two());

    data.iter_mut().for_each(|x| *x = x.conj());

    fft_rows(data, size);

    const N: usize = 32;
    utils::transpose_chunk::<_, N>(data, size);
    utils::transpose_blocks::<_, N>(data, size);

    fft_rows(data, size);

    let scale = 1.0 / (size * size) as f32;

    data.iter_mut().for_each(|x| *x = x.conj() * scale);
}

#[cfg(test)]
pub(crate) mod tests {
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

    pub(crate) fn approx_eq(a: &[Complex32], b: &[Complex32], eps: f32) {
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
                x.im,
                y.im
            );
        }
    }

    fn run_case(input: Box<[Complex32]>, eps: f32) {
        let mut mine = input.clone();
        fft_stockham(&mut mine);

        let reference = rustfft_reference(
            &input
                .iter()
                .map(|x| Complex32::new(x.re, x.im))
                .collect::<Vec<_>>(),
        );

        approx_eq(&mine, &reference, eps);
    }

    #[test]
    fn test_constant_one() {
        let data = [Complex32::ONE; 8];

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

        let mut rand_complex = || Complex32::new(rng.random(), rng.random());

        // 8
        {
            let mut arr = [Complex32::ZERO; 8];

            arr.fill_with(&mut rand_complex);

            run_case(arr.into(), EPSILON);
        }

        // 16
        {
            let mut arr = [Complex32::ZERO; 16];

            arr.fill_with(&mut rand_complex);

            run_case(arr.into(), EPSILON);
        }
    }

    #[test]
    fn test_impulse() {
        let mut arr = [Complex32::ZERO; 8];

        arr[0] = Complex32::new(1.0, 0.0);

        run_case(arr.into(), EPSILON);
    }

    #[test]
    fn test_constant() {
        let arr = [Complex32::new(2.0, 1.0); 16];

        run_case(arr.into(), EPSILON);
    }

    #[test]
    fn test_single_tone_sine() {
        const N: usize = 64;

        let freq = 5.0; // cycles over N samples

        let mut input = [Complex32::ZERO; N];

        for i in 0..N {
            let v = (TAU * freq * i as f32 / N as f32).sin();
            input[i] = Complex32::new(v, 0.0);
        }

        run_case(input.into(), EPSILON);
    }

    #[test]
    fn test_single_tone_cosine() {
        const N: usize = 64;
        let freq = 7.0;

        let mut input = [Complex32::ZERO; N];
        for i in 0..N {
            let v = (TAU * freq * i as f32 / N as f32).cos();
            input[i] = Complex32::new(v, 0.0);
        }

        run_case(input.into(), EPSILON);
    }

    #[test]
    fn test_two_tone_signal() {
        const N: usize = 128;
        let f1 = 3.0;
        let f2 = 17.0;

        let mut input = [Complex32::ZERO; N];

        for i in 0..N {
            let v = (TAU * f1 * i as f32 / N as f32).sin()
                + 0.5 * (TAU * f2 * i as f32 / N as f32).sin();
            input[i] = Complex32::new(v, 0.0);
        }

        run_case(input.into(), EPSILON);
    }

    #[test]
    fn test_chirp_signal() {
        const N: usize = 128;

        let mut input = [Complex32::ZERO; N];
        for i in 0..N {
            let f = 1.0 + 20.0 * (i as f32 / N as f32); // sweep from 1→21 cycles
            let v = (TAU * f * (i as f32 / N as f32)).sin();

            input[i] = Complex32::new(v, 0.0);
        }

        run_case(input.into(), EPSILON);
    }

    #[test]
    fn test_even_symmetric_real() {
        const N: usize = 32;

        let mut input = [Complex32::ZERO; N];

        for i in 0..N / 2 {
            let v = (i as f32).powi(2);
            input[i] = Complex32::new(v, 0.0);
            input[N - i - 1] = Complex32::new(v, 0.0);
        }

        run_case(input.into(), EPSILON);
    }

    #[test]
    fn test_odd_symmetric_real() {
        const N: usize = 32;

        let mut input = [Complex32::ZERO; N];

        for i in 0..N / 2 {
            let v = (i as f32).sin();

            input[i] = Complex32::new(v, 0.0);
            input[N - i - 1] = Complex32::new(-v, 0.0);
        }

        run_case(input.into(), EPSILON);
    }

    #[test]
    fn test_offset_impulse() {
        const N: usize = 64;

        let mut input = [Complex32::ZERO; N];

        input[7] = Complex32::new(1.0, 0.0);

        run_case(input.into(), EPSILON);
    }
}
