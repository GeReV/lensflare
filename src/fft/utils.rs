use cgmath::num_traits::NumAssign;
use num_complex::{Complex, Complex32};
use rustfft::num_traits::Float;

pub(crate) fn elementwise_multiply<T: Clone + NumAssign>(data: &mut [Complex<T>], kernel: &[Complex<T>]) {
    debug_assert_eq!(data.len(), kernel.len());

    data.iter_mut().zip(kernel).for_each(|(x, k)| {
        *x *= k;
    });
}

pub(crate) fn transpose_chunk<T, const N: usize>(buffer: &mut [T], buffer_width: usize) {
    let block_count = buffer_width / N;

    for block_y in 0..block_count {
        for block_x in 0..block_count {
            let block_offset = (block_y * N) * buffer_width + block_x * N;

            for y in 0..N {
                for x in y + 1..N {
                    let idx = block_offset + y * buffer_width + x;
                    let idx_t = block_offset + x * buffer_width + y;

                    buffer.swap(idx, idx_t);
                }
            }
        }
    }
}

pub(crate) fn transpose_blocks<T: Copy + Default, const N: usize>(buffer: &mut [T], buffer_width: usize) {
    let block_count = buffer_width / N;

    for block_y in 0..block_count {
        for block_x in (block_y + 1)..block_count {
            // Copy row-by-row.
            for y in 0..N {
                let src_offset = (block_y * N + y) * buffer_width + block_x * N;
                let dst_offset = (block_x * N + y) * buffer_width + block_y * N;

                let (src, dst) = buffer.split_at_mut(src_offset + N);

                let mut temp = [T::default(); N];
                temp.copy_from_slice(&src[src_offset..src_offset + N]);

                let split_dst_offset = dst_offset - src_offset - N;

                src[src_offset..src_offset + N]
                    .copy_from_slice(&dst[split_dst_offset..split_dst_offset + N]);

                dst[split_dst_offset..split_dst_offset + N].copy_from_slice(&temp);
            }
        }
    }
}

pub(crate) fn print_complex_slice(c: &[Complex32]) {
    for x in c {
        print!("{x:.3}; ");
    }
    println!();
}
