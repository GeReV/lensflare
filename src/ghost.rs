use crate::colors::wavelength_to_rgb;
use crate::fft::angular_spectrum::angular_spectrum;
use crate::fft::wgpu::{ComplexNormalizePipeline, ComputeFftPipeline, CopyToComplexPipeline, GenerateFrequenciesParameters, GenerateFrequenciesPipeline, MultiplyComplexPipeline, TextureMultiplyAddPipeline, TextureMultiplyConstPipeline};
use crate::shaders::create_shader;
use crate::texture::TextureExt;
use crate::vertex::Vertex;
use glam::Vec3;
use image::buffer::ConvertBuffer;
use image::{EncodableLayout, GrayImage, Pixel, Rgba, Rgba32FImage, RgbaImage};
use itertools::Itertools;
use num_complex::Complex32;
use rayon::prelude::*;
use std::f32::consts::TAU;
use wesl::{StandardResolver, Wesl};
use wgpu::util::{BufferInitDescriptor, DeviceExt, TextureBlitter, TextureBlitterBuilder};
use wgpu::wgt::TextureDescriptor;
use wgpu::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, BufferAddress, BufferUsages, Color,
    ColorTargetState, ColorWrites, CommandEncoderDescriptor, Device, Extent3d, FragmentState,
    FrontFace, IndexFormat, LoadOp, MultisampleState, Operations, PipelineCompilationOptions,
    PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology, Queue,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor, StoreOp, Texture,
    TextureDimension, TextureFormat, TextureUsages, VertexState,
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

    let view = texture.create_view_default();

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Render Ghost Encoder"),
    });

    {
        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Ghost Texture Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
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
        label: Some("Aperture Texture"),
        size: Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
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
            format: texture.format(),
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

    let view = texture.create_view_default();

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Render Ghost Encoder"),
    });

    {
        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Ghost Texture Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
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

pub fn fft_ghost(
    image: &GrayImage,
    delta: f32,
    z: f32,
    wavelengths: impl IntoIterator<Item = f32>,
) -> RgbaImage {
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

pub fn fft_ghost_gpu(
    device: &Device,
    queue: &Queue,
    compiler: &mut Wesl<StandardResolver>,
    aperture_texture: &Texture,
    delta: f32,
    z: f32,
    wavelengths: impl IntoIterator<Item = f32>,
) -> anyhow::Result<Texture> {
    let width = aperture_texture.width();
    let height = aperture_texture.height();

    let size = width as usize;

    let wavelengths = wavelengths.into_iter().collect_vec();

    let min_storage_buffer_offset_alignment =
        device.limits().min_storage_buffer_offset_alignment as usize;

    let colors = wavelengths
        .iter()
        .flat_map(|&x| {
            let mut bytes = vec![0u8; min_storage_buffer_offset_alignment];

            let color = wavelength_to_rgb(x)
                .clamp(Vec3::ZERO, Vec3::ONE)
                .extend(1.0);

            bytes[..size_of_val(&color)].copy_from_slice(bytemuck::cast_slice(&[color]));

            bytes
        })
        .collect::<Vec<_>>();

    let colors_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("FFT Colors Buffer"),
        contents: &colors,
        usage: BufferUsages::STORAGE,
    });

    let angular_spectrum_parameters =
        generate_angular_spectrum_parameters_buffer(delta, z, &wavelengths)
            .iter()
            .flat_map(|params| {
                let mut bytes = vec![0u8; min_storage_buffer_offset_alignment];

                bytes[..size_of_val(params)].copy_from_slice(bytemuck::cast_slice(&[*params]));

                bytes
            })
            .collect::<Vec<_>>();

    let angular_spectrum_parameters_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("FFT Angular Spectrum Parameters Buffer"),
        contents: &angular_spectrum_parameters,
        usage: BufferUsages::STORAGE,
    });

    let temp_textures_rg32float = (0..3)
        .map(|i| {
            device.create_texture(&TextureDescriptor {
                label: Some(&format!("Ghost Temporary Rg32Float Texture {i}")),
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                format: TextureFormat::Rg32Float,
                dimension: TextureDimension::D2,
                sample_count: 1,
                mip_level_count: 1,
                view_formats: &[],
            })
        })
        .collect_vec()
        .into_boxed_slice();

    let temp_texture_rg32float_views = temp_textures_rg32float
        .iter()
        .map(|tex| tex.create_view_default())
        .collect_vec()
        .into_boxed_slice();

    let temp_texture_r32float = device.create_texture(&TextureDescriptor {
        label: Some("FFT Temporary Texture"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
        dimension: TextureDimension::D2,
        format: TextureFormat::R32Float,
        sample_count: 1,
        mip_level_count: 1,
        view_formats: &[],
    });

    let temp_texture_r32float_view = temp_texture_r32float.create_view_default();

    let temp_textures_rgba32float = (0..2)
        .map(|i| {
            device.create_texture(&TextureDescriptor {
                label: Some(&format!("Ghost Temporary Rgba32Float Texture {i}")),
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                format: TextureFormat::Rgba32Float,
                dimension: TextureDimension::D2,
                sample_count: 1,
                mip_level_count: 1,
                view_formats: &[],
            })
        })
        .collect_vec()
        .into_boxed_slice();

    let mut temp_texture_rgba32float_views = temp_textures_rgba32float
        .iter()
        .map(|tex| tex.create_view_default())
        .collect_vec()
        .into_boxed_slice();

    let texture = device.create_texture(&TextureDescriptor {
        label: Some("Ghost Texture"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        usage: TextureUsages::TEXTURE_BINDING
            | TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::COPY_DST,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        sample_count: 1,
        mip_level_count: 1,
        view_formats: &[],
    });

    let texture_view = texture.create_view_default();

    let frequencies_texture = device.create_texture(&TextureDescriptor {
        label: Some("Angular Spectrum Frequencies Texture"),
        format: TextureFormat::Rg32Float,
        size: Extent3d {
            width: size as u32,
            height: size as u32,
            depth_or_array_layers: 1,
        },
        usage: TextureUsages::STORAGE_BINDING,
        dimension: TextureDimension::D2,
        mip_level_count: 1,
        sample_count: 1,
        view_formats: &[],
    });

    let frequencies_texture_view =
        frequencies_texture.create_view_default();

    let copy_to_complex_pipeline = CopyToComplexPipeline::new(device, compiler, size)?;

    let fft_pipeline = ComputeFftPipeline::new(device, compiler, size)?;

    let multiply_pipeline = MultiplyComplexPipeline::new(device, compiler, size)?;

    let generate_frequencies_pipeline =
        GenerateFrequenciesPipeline::new(device, compiler, size)?;

    let normalize_pipeline = ComplexNormalizePipeline::new(device, compiler)?;

    let texture_multiply_add = TextureMultiplyAddPipeline::new(device, compiler)?;

    let texture_multiply_constant = TextureMultiplyConstPipeline::new(
        device,
        compiler,
        TextureFormat::Rgba32Float,
        1.0 / wavelengths.len() as f32,
    )?;

    let mut command_encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Ghost FFT Command Encoder"),
    });

    let texture_blitter = TextureBlitter::new(device, texture.format());

    let texture_mask_blitter = TextureBlitterBuilder::new(device, texture.format())
        .blend_state(BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::Zero,
                dst_factor: BlendFactor::SrcAlpha,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::Zero,
                dst_factor: BlendFactor::SrcAlpha,
                operation: BlendOperation::Add,
            },
        })
        .build();

    let aperture_texture_view = aperture_texture.create_view_default();

    // Copy the aperture texture to the source buffer as complex numbers.
    copy_to_complex_pipeline.copy(
        device,
        &mut command_encoder,
        &aperture_texture_view,
        &temp_texture_rg32float_views[1],
    );

    fft_pipeline.process_fft2d(device, &mut command_encoder, &temp_texture_rg32float_views[1], &temp_texture_rg32float_views[0]);

    // Perform Angular Spectrum on a series of wavelengths and accumulate the result.
    for (i, _) in wavelengths.iter().enumerate() {
        generate_frequencies_pipeline.generate(
            device,
            &mut command_encoder,
            &frequencies_texture_view,
            &angular_spectrum_parameters_buffer,
            (i * min_storage_buffer_offset_alignment) as u32,
        );

        multiply_pipeline.multiply(
            device,
            &mut command_encoder,
            &temp_texture_rg32float_views[0],
            &frequencies_texture_view,
            &temp_texture_rg32float_views[1],
        );

        fft_pipeline
            .process_ifft2d(device, &mut command_encoder, &temp_texture_rg32float_views[1], &temp_texture_rg32float_views[2]);

        normalize_pipeline.normalize(
            device,
            &mut command_encoder,
            &temp_texture_rg32float_views[2],
            &temp_texture_r32float_view,
        );

        texture_multiply_add.process(
            device,
            &mut command_encoder,
            &temp_texture_rgba32float_views[0],
            &temp_texture_r32float_view,
            &temp_texture_rgba32float_views[1],
            &colors_buffer.slice((i * min_storage_buffer_offset_alignment) as BufferAddress..),
        );

        temp_texture_rgba32float_views.swap(0, 1);
    }

    // Average the accumulated result
    texture_multiply_constant.process(
        device,
        &mut command_encoder,
        &temp_texture_rgba32float_views[0],
        &temp_texture_rgba32float_views[1],
    );

    // Copy to final texture
    texture_blitter.copy(
        device,
        &mut command_encoder,
        &temp_texture_rgba32float_views[1],
        &texture_view,
    );

    // Multiply with aperture mask to trim away ringing outside the aperture shape
    texture_mask_blitter.copy(
        device,
        &mut command_encoder,
        &aperture_texture_view,
        &texture_view,
    );

    queue.submit(Some(command_encoder.finish()));

    Ok(texture)
}

fn generate_angular_spectrum_parameters_buffer(
    delta: f32,
    z: f32,
    wavelengths: &[f32],
) -> Vec<GenerateFrequenciesParameters> {
    wavelengths
        .iter()
        .map(|wavelength| GenerateFrequenciesParameters {
            delta,
            z,
            wavelength_meters: wavelength * 1e-9,
        })
        .collect()
}
