use crate::fft::fft_stockham;
use crate::fft::frft::frft_rows;
use crate::fft::utils::{transpose_blocks, transpose_chunk};
use num_complex::Complex32;
use std::f32::consts::PI;

pub fn fresnel_diffraction_image(
    data: &mut [Complex32],
    size: u32,
    alpha: f32,
    z: f32,
    wavelength: f32,
    padding_factor: usize,
) {
    assert!(size.is_power_of_two());

    let w = size as isize;
    let pw = w / 2 * padding_factor as isize;
    let hw = w / 2;

    let chirp = (-hw..hw)
        .into_iter()
        .map(|k| Complex32::cis(PI * alpha * (k * k) as f32))
        .collect::<Vec<_>>();

    let mut kernel = (-pw..pw)
        .into_iter()
        .map(|k| Complex32::cis(PI * alpha * (k * k) as f32))
        .collect::<Vec<_>>();

    // Taper h (time-domain) before FFT
    // Hann window of length 2N: w[j] = 0.5*(1 - cos(2π j/(2N-1)))
    kernel
        .iter_mut()
        .enumerate()
        .for_each(|(j, x)| *x *= 0.5 * (1.0 - (2.0 * PI * j as f32 / (2 * size - 1) as f32).cos()));

    // FFT of the chirp (h[n]).
    fft_stockham(&mut kernel);

    frft_rows(data, &chirp, &chirp, &kernel, size as usize, padding_factor);

    const N: usize = 32;
    transpose_chunk::<_, N>(data, size as usize);
    transpose_blocks::<_, N>(data, size as usize);

    frft_rows(data, &chirp, &chirp, &kernel, size as usize, padding_factor);

    let k = 2.0 * PI / wavelength;
    let prefactor = Complex32::cis(k * z);

    data.iter_mut().for_each(|x| *x /= prefactor);
}
