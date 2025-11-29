use crate::colors::wavelength_to_rgb;
use crate::fft::angular_spectrum::angular_spectrum;
use crate::shaders::create_shader;
use crate::vertex::Vertex;
use glam::Vec3;
use image::buffer::ConvertBuffer;
use image::{DynamicImage, EncodableLayout, GrayImage, Pixel, Rgba, Rgba32FImage, RgbaImage};
use itertools::Itertools;
use num_complex::Complex32;
use rayon::prelude::*;
use std::f32::consts::TAU;
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

pub fn draw_ghost_polygon(
    device: &Device,
    queue: &Queue,
    compiler: &mut Wesl<StandardResolver>,
    size: u32,
    polygon_sides: usize,
    polygon_radius: f32,
    polygon_rotation: f32,
) -> anyhow::Result<Texture> {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("Ghost Texture"),
        size: Extent3d {
            width: size,
            height: size,
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
        "package::ghost_polygon",
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

pub fn draw_ghost_sdf(
    device: &Device,
    queue: &Queue,
    compiler: &mut Wesl<StandardResolver>,
    size: u32,
    polygon_sides: usize,
    polygon_radius: f32,
    polygon_rotation: f32,
) -> anyhow::Result<Texture> {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("Ghost Texture"),
        size: Extent3d {
            width: size,
            height: size,
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

    compiler.add_constants([
        ("polygon_sides", polygon_sides as f64),
        ("polygon_radius", polygon_radius as f64),
        ("polygon_rotation", polygon_rotation as f64),
    ]);

    let shader = create_shader(
        device,
        compiler,
        Some("Render Ghost Shader"),
        "package::ghost_sdf",
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
                buffers: &[],
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

        render_pass.set_pipeline(&render_pipeline);

        render_pass.draw(0..3, 0..1);
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

pub fn fft_ghost(image: &GrayImage, delta: f32, z: f32, wavelengths: impl IntoIterator<Item = f32>) -> RgbaImage {
    let (width, height) = image.dimensions();

    let mut img = Rgba32FImage::new(width, height);

    let data = image
        .as_bytes()
        .iter()
        .map(|&x| Complex32::new((x as f32) / 255.0, 0.0))
        .collect_vec();

    let wavelengths = wavelengths.into_iter().collect_vec();

    for &wavelength in &wavelengths {
        let mut data = data.clone();

        let wavelength_meters = wavelength * 1e-9;

        angular_spectrum(&mut data, width as usize, delta, z, wavelength_meters);

        let values = data.iter().map(|x| x.norm()).collect::<Vec<_>>();

        let color = wavelength_to_rgb(wavelength);

        img.par_pixels_mut().zip(values).for_each(|(base, v)| {
            let pixel = Rgba((color * v).extend(v).to_array());

            base.apply2(&pixel, |a, b| a + b);
        });
    }

    let l = wavelengths.len() as f32;

    img.par_pixels_mut().for_each(|pixel| {
        pixel.apply(|p| p / l);
    });

    let img: RgbaImage = img.convert();

    img
}
