use crate::fft::wgpu::{
    ComplexNormalizePipeline, ComputeFftPipeline, CopyToComplexPipeline,
    TextureMultiplyConstPipeline,
};
use crate::shaders::create_shader;
use crate::texture::TextureExt;
use crate::utils::{wavelengths_to_colors, ADDITIVE_BLEND};
use glam::Vec4;
use itertools::Itertools;
use wesl::{StandardResolver, Wesl};
use wgpu::util::{BufferInitDescriptor, DeviceExt, TextureBlitter};
use wgpu::wgt::{SamplerDescriptor, TextureDescriptor};
use wgpu::{
    AddressMode, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry,
    BindingResource, BindingType, Buffer, BufferBindingType, BufferUsages, Color, ColorTargetState,
    ColorWrites, CommandEncoder, CommandEncoderDescriptor, Device, Extent3d, Face, FilterMode,
    FragmentState, FrontFace, LoadOp, MultisampleState, Operations, PipelineCompilationOptions,
    PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology, Queue,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    SamplerBindingType, ShaderStages, StoreOp, Texture, TextureDimension, TextureFormat,
    TextureSampleType, TextureUsages, TextureView, TextureViewDimension, VertexState,
};

pub fn create_starburst_gpu(
    device: &Device,
    queue: &Queue,
    compiler: &mut Wesl<StandardResolver>,
    aperture_texture: &Texture,
    wavelengths: &[f32],
) -> anyhow::Result<Texture> {
    let width = aperture_texture.width();
    let height = aperture_texture.height();

    let size = width as usize;

    let min_storage_buffer_offset_alignment =
        device.limits().min_storage_buffer_offset_alignment as usize;

    let colors = wavelengths_to_colors(wavelengths)
        .iter()
        .flat_map(|&color| {
            let mut bytes = vec![0u8; min_storage_buffer_offset_alignment];

            bytes[..size_of_val(&color)].copy_from_slice(bytemuck::cast_slice(&[color]));

            bytes
        })
        .collect::<Vec<_>>();

    let colors_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Starburst Colors Buffer"),
        contents: &colors,
        usage: BufferUsages::STORAGE,
    });

    let texture = device.create_texture(&TextureDescriptor {
        label: Some("Starburst Texture"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        sample_count: 1,
        mip_level_count: 1,
        view_formats: &[],
    });

    let texture_view = texture.create_view_default();

    let temp_textures_rg32float = (0..2)
        .map(|i| {
            device.create_texture(&TextureDescriptor {
                label: Some(&format!("Starburst Temporary Rg32Float Texture {i}")),
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
        label: Some("Starburst FFT Temporary R32Float Texture"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        usage: TextureUsages::TEXTURE_BINDING
            | TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::STORAGE_BINDING,
        dimension: TextureDimension::D2,
        format: TextureFormat::R32Float,
        sample_count: 1,
        mip_level_count: 1,
        view_formats: &[],
    });
    let temp_texture_r32float_view = temp_texture_r32float.create_view_default();

    let temp_texture_r16float = device.create_texture(&TextureDescriptor {
        label: Some("Starburst FFT Temporary R16Float Texture"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
        dimension: TextureDimension::D2,
        format: TextureFormat::R16Float,
        sample_count: 1,
        mip_level_count: 1,
        view_formats: &[],
    });
    let temp_texture_r16float_view = temp_texture_r16float.create_view_default();

    let copy_to_complex_pipeline = CopyToComplexPipeline::new(device, compiler, size)?;

    let fft_pipeline = ComputeFftPipeline::new(device, compiler, size)?;

    let normalize_pipeline = ComplexNormalizePipeline::new(device, compiler)?;

    let texture_multiply_const =
        TextureMultiplyConstPipeline::new(device, compiler, TextureFormat::Rg32Float, size as f32)?;

    let starburst_pipeline = StarburstPipeline::new(device, compiler)?;

    let texture_blitter = TextureBlitter::new(device, TextureFormat::R16Float);

    let mut command_encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Starburst Command Encoder"),
    });

    // Copy the aperture texture to the source buffer as complex numbers.
    copy_to_complex_pipeline.copy(
        device,
        &mut command_encoder,
        &aperture_texture.create_view_default(),
        &temp_texture_rg32float_views[0],
    );

    fft_pipeline.process_fft2d(
        device,
        &mut command_encoder,
        &temp_texture_rg32float_views[0],
        &temp_texture_rg32float_views[1],
    );

    texture_multiply_const.process(
        device,
        &mut command_encoder,
        &temp_texture_rg32float_views[1],
        &temp_texture_rg32float_views[0],
    );

    normalize_pipeline.normalize(
        device,
        &mut command_encoder,
        &temp_texture_rg32float_views[0],
        &temp_texture_r32float_view,
    );

    texture_blitter.copy(
        device,
        &mut command_encoder,
        &temp_texture_r32float_view,
        &temp_texture_r16float_view,
    );

    starburst_pipeline.render(
        device,
        &mut command_encoder,
        &temp_texture_r16float_view,
        &texture_view,
        &colors_buffer,
    );

    queue.submit(Some(command_encoder.finish()));

    Ok(texture)
}

struct StarburstPipeline {
    pipeline: RenderPipeline,
    texture_bind_group_layout: BindGroupLayout,
    colors_bind_group_layout: BindGroupLayout,
}

impl StarburstPipeline {
    pub fn new(device: &Device, compiler: &mut Wesl<StandardResolver>) -> anyhow::Result<Self> {
        compiler.add_constants([("base_size", 1.0), ("min_size", 0.5)]);

        let module = create_shader(
            device,
            compiler,
            Some("Starburst Render Shader"),
            "package::starburst",
        )?;

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Starburst Texture Bind Group Layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let colors_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Starburst Colors Bind Group Layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Starburst Render Pipeline Layout"),
            bind_group_layouts: &[&texture_bind_group_layout, &colors_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Starburst Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend: Some(ADDITIVE_BLEND),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Cw,
                cull_mode: Some(Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Ok(Self {
            pipeline,
            texture_bind_group_layout,
            colors_bind_group_layout,
        })
    }

    pub fn render(
        &self,
        device: &Device,
        command_encoder: &mut CommandEncoder,
        src: &TextureView,
        dst: &TextureView,
        colors: &Buffer,
    ) {
        debug_assert!(
            src.texture()
                .usage()
                .contains(TextureUsages::TEXTURE_BINDING),
            "Source texture must be TEXTURE_BINDING usage"
        );
        debug_assert!(
            dst.texture()
                .usage()
                .contains(TextureUsages::RENDER_ATTACHMENT),
            "Destination texture must be RENDER_ATTACHMENT usage"
        );

        let mut render_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Starburst Render Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: dst,
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

        let starburst_texture_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("Starburst Texture Sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });

        let texture_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Starburst Texture Bind Group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&starburst_texture_sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(src),
                },
            ],
        });

        let colors_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Starburst Colors Bind Group"),
            layout: &self.colors_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: colors.as_entire_binding(),
            }],
        });

        render_pass.set_pipeline(&self.pipeline);

        render_pass.set_bind_group(0, &texture_bind_group, &[]);
        render_pass.set_bind_group(1, &colors_bind_group, &[]);

        let instance_count = colors.size() as usize / size_of::<Vec4>();

        render_pass.draw(0..3, 0..instance_count as u32);
    }
}
