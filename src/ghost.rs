use crate::fft::complex::Complex;
use crate::fft::{fft_stockham, fftshift};
use crate::shaders::create_shader;
use crate::vertex::Vertex;
use glam::Vec3;
use itertools::Itertools;
use rayon::prelude::*;
use std::f32::consts::TAU;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;
use wesl::{StandardResolver, Wesl};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::wgt::{TextureDescriptor, TextureViewDescriptor};
use wgpu::{
    BlendState, BufferAddress, BufferDescriptor, BufferUsages, Color, ColorTargetState,
    ColorWrites, CommandEncoderDescriptor, Device, Extent3d, FragmentState, FrontFace, IndexFormat,
    LoadOp, MultisampleState, Operations, PipelineCompilationOptions, PipelineLayoutDescriptor,
    PolygonMode, PrimitiveState, PrimitiveTopology, Queue, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipelineDescriptor, StoreOp, Texture, TextureDimension,
    TextureFormat, TextureUsages, VertexState,
};

pub fn draw_ghost(
    device: &Device,
    queue: &Queue,
    compiler: &mut Wesl<StandardResolver>,
    polygon_sides: usize,
    polygon_radius: f32,
    polygon_rotation: f32,
) -> anyhow::Result<Texture> {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("Ghost Texture"),
        size: Extent3d {
            width: 1024,
            height: 1024,
            depth_or_array_layers: 1,
        },
        dimension: TextureDimension::D2,
        format: TextureFormat::R8Unorm,
        sample_count: 1,
        mip_level_count: 1,
        usage: TextureUsages::TEXTURE_BINDING
            | TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let shader = create_shader(
        device,
        compiler,
        Some("Render Ghost Shader"),
        "package::ghost",
    )?;

    let render_pipeline = {
        let render_pipeline_lines_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Render Ghost Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let targets = [Some(ColorTargetState {
            format: TextureFormat::R8Unorm,
            blend: Some(BlendState::REPLACE),
            write_mask: ColorWrites::ALL,
        })];

        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render Ghost Pipeline"),
            layout: Some(&render_pipeline_lines_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &targets,
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Cw,
                cull_mode: None,
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            depth_stencil: None,
            multiview: None,
            cache: None,
        })
    };

    let (vertices, indices) = generate_polygon(polygon_sides, polygon_radius, polygon_rotation);

    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Ghost Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Ghost Index Buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: BufferUsages::INDEX,
    });

    let view = texture.create_view(&TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Render Ghost Encoder"),
    });

    {
        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Ghost Texture Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::TRANSPARENT),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_index_buffer(index_buffer.slice(..), IndexFormat::Uint32);

        render_pass.set_pipeline(&render_pipeline);

        render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
    }

    queue.submit(Some(encoder.finish()));

    Ok(texture)
}

fn generate_polygon(sides: usize, radius: f32, rotation: f32) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(sides + 1);
    let mut indices = Vec::with_capacity(sides * 3);

    vertices.push(Vertex {
        position: Vec3::ZERO,
    });

    // Start from the top and rotate counter-clockwise the specified amount.
    let mut start = TAU * 0.25 - rotation;

    // Rotate even-sided polygons so it sits on a flat base.
    start += if sides.is_multiple_of(2) {
        TAU / sides as f32 * 0.5
    } else {
        0.0
    };

    for i in 0..sides {
        let angle = start + TAU * i as f32 / sides as f32;
        let (y, x) = angle.sin_cos();

        vertices.push(Vertex {
            position: Vec3::new(x, y, 0.0) * radius,
        });
    }

    for i in 1..=sides {
        indices.push(0);
        indices.push((i % sides + 1) as u32);
        indices.push(i as u32);
    }

    (vertices, indices)
}

pub fn test_ghost(device: &Device, queue: &Queue, ghost_texture: &Texture) {
    let ghost_texture_size = ghost_texture.size();

    let output_buffer_size =
        (ghost_texture_size.width * ghost_texture_size.height) as BufferAddress;
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
            texture: ghost_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ghost_texture_size.width),
                rows_per_image: Some(ghost_texture_size.height),
            },
        },
        ghost_texture_size,
    );

    queue.submit(Some(encoder.finish()));

    {
        let mut texture_data = Vec::with_capacity(output_buffer_size as usize);

        let buffer_slice = output_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
        device.poll(wgpu::PollType::wait()).unwrap();
        receiver.recv().unwrap().unwrap();
        {
            let view = buffer_slice.get_mapped_range();
            texture_data.extend_from_slice(&view[..]);
        }
        output_buffer.unmap();

        let Extent3d { width, height, .. } = ghost_texture_size;

        fft_image(&mut texture_data, width, height);
    }
}

fn fft_image(data: &mut [u8], width: u32, height: u32) {
    image::GrayImage::from_raw(width, height, data.to_vec())
        .unwrap()
        .save("temp0.png")
        .unwrap();

    let mut data = data
        .iter()
        .map(|&x| Complex::new((x as f32) / 255.0, 0.0))
        .collect_vec();

    fft_rows(&mut data, width as usize);

    // image::GrayImage::from_raw(width, height, data.to_vec())
    //     .unwrap()
    //     .save("temp1.png")
    //     .unwrap();

    const N: usize = 32;
    transpose_chunk::<_, N>(&mut data, width as usize);
    transpose_blocks::<_, N>(&mut data, width as usize);

    // image::GrayImage::from_raw(width, height, data.to_vec())
    //     .unwrap()
    //     .save("temp2.png")
    //     .unwrap();

    fft_rows(&mut data, width as usize);

    // image::GrayImage::from_raw(width, height, data.to_vec())
    //     .unwrap()
    //     .save("temp3.png")
    //     .unwrap();

    fftshift(&mut data, width as usize);

    image::GrayImage::from_raw(
        width,
        height,
        data.iter().map(|x| x.re as u8).collect(),
    )
    .unwrap()
    .save("temp4.png")
    .unwrap();
}

fn fft_rows(data: &mut [Complex], width: usize) {
    data.par_chunks_mut(width).for_each(fft_stockham);
}

fn transpose_chunk<T, const N: usize>(buffer: &mut [T], buffer_width: usize) {
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

fn transpose_blocks<T: Copy + Default, const N: usize>(buffer: &mut [T], buffer_width: usize) {
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
