use crate::fft::utils::elementwise_multiply;
use crate::fft::{fft2d, ifft2d};
use num_complex::{Complex, Complex32};
use rayon::prelude::*;

pub(crate) fn generate_frequency_grid(
    size: usize,
    delta: f32,
    z: f32,
    wavelength: f32,
) -> Vec<Complex32> {
    use std::f32::consts::PI;

    let pw = size * 2;
    let k = 2.0 * PI / wavelength;

    let mut frequency_grid = Vec::with_capacity(pw * pw);

    for u in 0..pw {
        let fu = (u as isize - size as isize) as f32 / (delta * pw as f32);
        let kx = 2.0 * PI * fu;

        for v in 0..pw {
            let fv = (v as isize - size as isize) as f32 / (delta * pw as f32);
            let ky = 2.0 * PI * fv;
            let kz2 = k * k - (kx * kx + ky * ky);

            let kz = if kz2 >= 0.0 {
                Complex::new(kz2.sqrt(), 0.0)
            } else {
                Complex::new(0.0, -(-kz2).sqrt())
            };

            frequency_grid.push((Complex32::i() * kz * z).exp());
        }
    }

    frequency_grid
}

// pub(crate) fn generate_frequency_grid(size: usize, delta: f32, z: f32, wavelength: f32) -> Vec<Complex32> {
//     use std::f32::consts::PI;
//
//     let pw = size * 2;
//
//     let k = 1.0 / (wavelength * wavelength);
//
//     let mut frequency_grid = Vec::with_capacity(pw * pw);
//
//     let step = 1.0 / (delta * size as f32);
//
//     for u in 0..pw {
//         let kx = (u as isize - size as isize) as f32 * step;
//         let w_x = 2.0 * z * step;
//         let halfband_x = 1.0 / ((w_x * w_x + 1.0).sqrt() * wavelength);
//
//         for v in 0..pw {
//             let ky = (v as isize - size as isize) as f32 * step;
//             let kz2 = k - (kx * kx + ky * ky);
//
//             let w_y = 2.0 * z * step;
//             let halfband_y = 1.0 / ((w_y * w_y + 1.0).sqrt() * wavelength);
//
//             let value = if kz2 > 0.0 && kx.abs() < halfband_x && ky.abs() < halfband_y {
//                 Complex32::cis(2.0 * PI * kz2.sqrt() * z)
//             } else {
//                 Complex32::ZERO
//             };
//
//             frequency_grid.push(value);
//         }
//     }
//
//     frequency_grid
// }

// TODO: Does changing the copying/cropping code result in better results?
pub(crate) fn angular_spectrum(
    data: &mut [Complex32],
    size: usize,
    delta: f32,
    z: f32,
    wavelength_meters: f32,
) {
    assert!(size.is_power_of_two());
    assert_eq!(data.len(), size * size);

    let pw = size * 2;
    let start = ((pw as isize - size as isize) / 2) as usize;

    let mut padded_data = vec![Complex32::ZERO; pw * pw];

    let frequency_grid = generate_frequency_grid(size, delta, z, wavelength_meters);

    // Copy the input data to the center of the padded grid, row by row.
    {
        // Skip the first `start` rows of the padded grid.
        let (_, padded_start) = padded_data.split_at_mut(start * pw);

        let padded_data_middle_chunks = padded_start[..size * pw].par_chunks_exact_mut(pw);

        // Copy data row by row.
        data.par_chunks_exact_mut(size)
            .zip(padded_data_middle_chunks)
            .for_each(|(src_chunk, dest_chunk)| {
                dest_chunk[start..start + size].copy_from_slice(src_chunk);
            });
    }

    fft2d(&mut padded_data, pw);

    elementwise_multiply(&mut padded_data, &frequency_grid);

    ifft2d(&mut padded_data, pw);

    // Crop the result to the original size.
    {
        // Skip the first `start` rows of the padded grid.
        let (_, padded_start) = padded_data.split_at_mut(start * pw);

        let padded_data_middle_chunks = padded_start[..size * pw].par_chunks_exact_mut(pw);

        // Copy data to the center of the padded grid, row by row.
        padded_data_middle_chunks
            .zip(data.par_chunks_exact_mut(size))
            .for_each(|(src_chunk, dest_chunk)| {
                dest_chunk.copy_from_slice(&src_chunk[start..start + size]);
            });
    }
}

