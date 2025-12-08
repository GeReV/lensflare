use cgmath::num_traits::NumAssign;
use image::{DynamicImage, GrayImage, Luma, RgbaImage};
use itertools::Itertools;
use num_complex::{Complex, Complex32};
use wgpu::{Buffer, BufferAddress, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Device, Queue, Texture, TextureFormat};

pub(crate) fn elementwise_multiply<T: Clone + NumAssign>(
    data: &mut [Complex<T>],
    kernel: &[Complex<T>],
) {
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

pub(crate) fn transpose_blocks<T: Copy + Default, const N: usize>(
    buffer: &mut [T],
    buffer_width: usize,
) {
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

pub fn copy_buffer_from_gpu(device: &Device, buffer: &Buffer) -> Vec<u8> {
    let mut buffer_data = Vec::with_capacity(buffer.size() as usize);

    let buffer_slice = buffer.slice(..);

    let (sender, receiver) = std::sync::mpsc::channel();

    buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    receiver.recv().unwrap().unwrap();

    {
        let view = buffer_slice.get_mapped_range();
        buffer_data.extend_from_slice(&view[..]);
    }

    buffer.unmap();

    buffer_data
}

pub fn copy_texture_to_image(device: &Device, queue: &Queue, texture: &Texture) -> DynamicImage {
    let texture_size = texture.size();

    let bpp = texture.format().target_pixel_byte_cost().unwrap_or(1);

    let output_buffer_size = (texture_size.width * texture_size.height * bpp) as BufferAddress;
    let output_buffer = device.create_buffer(&BufferDescriptor {
        size: output_buffer_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        label: None,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            aspect: wgpu::TextureAspect::All,
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(texture_size.width * bpp),
                rows_per_image: Some(texture_size.height),
            },
        },
        texture_size,
    );

    queue.submit(Some(encoder.finish()));

    let texture_data = copy_buffer_from_gpu(device, &output_buffer);

    match texture.format() {
        TextureFormat::R8Unorm
        | TextureFormat::R8Snorm
        | TextureFormat::R8Uint
        | TextureFormat::R8Sint => DynamicImage::ImageLuma8(
            GrayImage::from_raw(texture_size.width, texture_size.height, texture_data).unwrap(),
        ),
        TextureFormat::Rgba8UnormSrgb
        | TextureFormat::Rgba8Unorm
        | TextureFormat::Rgba8Uint
        | TextureFormat::Rgba8Sint
        | TextureFormat::Rgba8Snorm => DynamicImage::ImageRgba8(
            RgbaImage::from_raw(texture_size.width, texture_size.height, texture_data).unwrap(),
        ),
        _ => unimplemented!("Unsupported texture format"),
    }
}

pub(crate) fn save_complex_data_to_image(data: &[Complex32], size: u32, name: impl Into<String>) {
    let real = GrayImage::from_par_fn(size, size, |x, y| {
        let value = data[(y * size + x) as usize];

        Luma([(value.re * 255.0) as u8])
    });

    let imaginary = GrayImage::from_par_fn(size, size, |x, y| {
        let value = data[(y * size + x) as usize];

        Luma([(value.im * 255.0) as u8])
    });

    let combined = GrayImage::from_par_fn(size, size, |x, y| {
        let value = data[(y * size + x) as usize];

        Luma([(value.norm() * 255.0) as u8])
    });

    let name = name.into();

    real.save(format!("{}_re.png", name)).unwrap();
    imaginary.save(format!("{}_im.png", name)).unwrap();
    combined.save(format!("{}.png", name)).unwrap();
}
