use std::f32::consts::PI;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use itertools::Itertools;
use num_complex::{Complex32, ComplexFloat};
use rayon::prelude::ParallelSliceMut;
use crate::fft::fft_stockham;
use crate::fft::utils::{elementwise_multiply, print_complex_slice, transpose_blocks, transpose_chunk};



fn chirp_coefficient(a: f32, k: isize, n: isize) -> Complex32 {
    Complex32::cis(PI * a * (k * k) as f32 / n as f32)
}

fn frft_generate_chirps(alpha: f32, n: usize) -> (Box<[Complex32]>, Box<[Complex32]>) {
    let (alpha_sin, alpha_cos) = alpha.sin_cos();
    let alpha_cot = alpha_cos / alpha_sin;

    let n = n as isize;
    let half = n / 2;

    let chirp_pre = (-half..half)
        .into_par_iter()
        .map(|k| {
            chirp_coefficient(alpha_cot, k, n)
        })
        .collect::<Vec<_>>();

    let chirp_post = (-half..half)
        .into_par_iter()
        .map(|k| {
            chirp_coefficient(alpha_cot, k, n)
        })
        .collect::<Vec<_>>();

    (chirp_pre.into_boxed_slice(), chirp_post.into_boxed_slice())
}

pub(crate) fn frft_generate_kernel(alpha: f32, n: usize) -> Box<[Complex32]> {
    let alpha_csc = 1.0 / alpha.sin();

    let n = n as isize;

    let kernel = (-n..n)
        .into_par_iter()
        .map(|k| chirp_coefficient(-alpha_csc, k, n))
        .collect::<Vec<_>>();

    kernel.into_boxed_slice()
}

pub(crate) fn frft_rows(data: &mut [Complex32], chirp_pre: &[Complex32], chirp_post: &[Complex32], kernel: &[Complex32], size: usize, padding_factor: usize) {
    data.par_chunks_mut(size).for_each(|chunk| {
        frft_step(chunk, chirp_pre, chirp_post, kernel, size, padding_factor);
    });
}

fn frft_step(chunk: &mut [Complex32], chirp_pre: &[Complex32], chirp_post: &[Complex32], kernel: &[Complex32], size: usize, padding_factor: usize) {
    // Multiply f[n] by the chirp.
    elementwise_multiply(chunk, chirp_pre);

    let mut padded_chunk = vec![Complex32::ZERO; size * padding_factor];
    padded_chunk[..size].copy_from_slice(chunk);

    // FFT of g[n].
    fft_stockham(&mut padded_chunk);

    // Multiply g[n] by h[n] to get y[n].
    elementwise_multiply(&mut padded_chunk, kernel);

    // IFFT of y[n].
    {
        padded_chunk.iter_mut().for_each(|x| {
            *x = x.conj();
        });

        fft_stockham(&mut padded_chunk);

        let scale = (2.0 * size as f32 * padding_factor as f32).recip();

        padded_chunk.iter_mut().for_each(|x| {
            *x = x.conj() * scale;
        });
    }

    let mid = (size * padding_factor) / 2;
    chunk.copy_from_slice(&padded_chunk[mid..mid + size]);

    // Multiply y[n] by the chirp again.
    elementwise_multiply(chunk, chirp_post);
}

pub fn frft_image(data: &mut [u8], width: u32, height: u32, padding_factor: usize) {
    assert!(width.is_power_of_two());

    let alpha = 0.15 * (600.0 / 400.0) * (2.0.sqrt().powf(5.0) / 18.0);
    let alpha = (1.0 - alpha) * 0.5 * PI;

    let (alpha_sin, alpha_cos) = alpha.sin_cos();
    let alpha_cot = alpha_cos / alpha_sin;
    let alpha_csc = 1.0 / alpha_sin;

    let w = width as isize;
    let hw = w / 2;

    let (chirp_pre, chirp_post) = frft_generate_chirps(alpha, w as usize);

    let mut kernel = frft_generate_kernel(alpha, w as usize);

    // FFT of the chirp (h[n]).
    fft_stockham(&mut kernel);

    let mut complex = data
        .iter()
        .map(|&x| Complex32::new((x as f32) / 255.0, 0.0))
        .collect_vec();

    frft_rows(&mut complex, &chirp_pre, &chirp_post, &kernel, width as usize, padding_factor);

    const N: usize = 32;
    transpose_chunk::<_, N>(&mut complex, width as usize);
    transpose_blocks::<_, N>(&mut complex, width as usize);

    frft_rows(&mut complex, &chirp_pre, &chirp_post, &kernel, width as usize, padding_factor);

    // fftshift(&mut Complex32, width as usize);

    // let c = Complex32::new(1.0, -alpha_cot).sqrt();
    //
    // complex.par_iter_mut().for_each(|x| *x *= c);

    let bytes = complex.iter().map(|x| x.norm() as u8).collect::<Vec<_>>();

    data.copy_from_slice(&bytes);

    image::GrayImage::from_raw(width, height, bytes)
        .unwrap()
        .save("temp1.png")
        .unwrap();
}

#[cfg(test)]
mod tests {
    use crate::fft::tests::approx_eq;
    use super::*;

    #[test]
    fn test_chirp_coefficients() {
        let alpha = PI * 0.5;
        let (alpha_sin, alpha_cos) = alpha.sin_cos();
        let alpha_cot = alpha_cos / alpha_sin;

        assert_eq!(chirp_coefficient(alpha_cot, 0, 8), Complex32::ONE);
    }

    #[test]
    fn test_impulse() {
        let mut data  = [Complex32::ZERO; 8];
        data[4] = Complex32::ONE;

        let alpha = PI * 0.5;

        let w = 8;

        let (chirp_pre, chirp_post) = frft_generate_chirps(alpha, w);

        let kernel = frft_generate_kernel(alpha, w);

        let mut kernel_h = kernel.clone();
        fft_stockham(&mut kernel_h);

        // Verify known result for G[k].
        {
            let mut expected = data;
            elementwise_multiply(&mut expected, &chirp_pre);

            fft_stockham(&mut expected);

            let c = Complex32::ONE;
            approx_eq(&expected, &[c, -c, c, -c, c, -c, c, -c], 1e-3);
        }

        // Check that the IFFT result is kernel shifted by half the width.
        {
            // Multiply f[n] by the chirp.
            elementwise_multiply(&mut data, &chirp_pre);

            let mut padded_chunk = vec![Complex32::ZERO; w * 2];
            padded_chunk[..w].copy_from_slice(&data);

            // FFT of g[n].
            fft_stockham(&mut padded_chunk);

            // Multiply g[n] by h[n] to get y[n].
            elementwise_multiply(&mut padded_chunk, &kernel_h);

            // IFFT of y[n].
            {
                padded_chunk.iter_mut().for_each(|x| {
                    *x = x.conj();
                });

                fft_stockham(&mut padded_chunk);

                let scale = (2.0 * w as f32).recip();

                padded_chunk.iter_mut().for_each(|x| {
                    *x = x.conj() * scale;
                });
            }

            padded_chunk.rotate_right(w / 2);

            approx_eq(&padded_chunk, &kernel, 1e-3);
        }
    }
}