use wgpu::{AddressMode, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource, BindingType, BlendState, Buffer, BufferBindingType, Color, ColorTargetState, ColorWrites, CommandEncoder, Device, Face, FilterMode, FragmentState, FrontFace, LoadOp, MultisampleState, Operations, PipelineCompilationOptions, PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology, RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, SamplerBindingType, ShaderStages, StoreOp, TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDimension, VertexState};
use wesl::{StandardResolver, Wesl};
use wgpu::wgt::SamplerDescriptor;
use crate::shaders::create_shader;

pub struct StarburstPipeline {
    starburst_pipeline: RenderPipeline,
    texture_bind_group_layout: BindGroupLayout,
    colors_bind_group_layout: BindGroupLayout,
    starburst_filter_pipeline: RenderPipeline,
}

impl StarburstPipeline {
    pub fn new(device: &Device, compiler: &mut Wesl<StandardResolver>) -> anyhow::Result<Self> {
        compiler.add_constants([("base_size", 1.0), ("min_size", 0.5)]);

        let module_starburst = create_shader(
            device,
            compiler,
            Some("Starburst Render Shader"),
            "package::starburst",
        )?;

        let module_filter = create_shader(
            device,
            compiler,
            Some("Starburst Render Shader"),
            "package::starburst_filter",
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
                    visibility: ShaderStages::VERTEX_FRAGMENT,
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
            bind_group_layouts: &[
                &texture_bind_group_layout,
                &texture_bind_group_layout,
                &colors_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let pipeline_descriptor = RenderPipelineDescriptor {
            label: Some("Starburst Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &module_starburst,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(FragmentState {
                module: &module_starburst,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend: Some(BlendState::REPLACE),
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
        };

        let starburst_pipeline = device.create_render_pipeline(&pipeline_descriptor);

        let starburst_filter_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Starburst Filter Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout, &colors_bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline_descriptor = RenderPipelineDescriptor {
            label: Some("Starburst Filter Pipeline"),
            layout: Some(&starburst_filter_pipeline_layout),
            fragment: Some(FragmentState {
                module: &module_filter,
                ..pipeline_descriptor.fragment.unwrap()
            }),
            ..pipeline_descriptor
        };
        
        let starburst_filter_pipeline = device.create_render_pipeline(&pipeline_descriptor);

        Ok(Self {
            starburst_pipeline,
            starburst_filter_pipeline,
            texture_bind_group_layout,
            colors_bind_group_layout,
        })
    }

    pub fn render(
        &self,
        device: &Device,
        command_encoder: &mut CommandEncoder,
        src: &TextureView,
        dust: &TextureView,
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

        let texture_sampler = device.create_sampler(&SamplerDescriptor {
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
                    resource: BindingResource::Sampler(&texture_sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(src),
                },
            ],
        });

        let dust_texture_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Dust Texture Bind Group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&texture_sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(dust),
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

        render_pass.set_pipeline(&self.starburst_pipeline);

        render_pass.set_bind_group(0, &texture_bind_group, &[]);
        render_pass.set_bind_group(1, &dust_texture_bind_group, &[]);
        render_pass.set_bind_group(2, &colors_bind_group, &[]);

        render_pass.draw(0..3, 0..1);
    }

    pub fn filter(
        &self,
        device: &Device,
        command_encoder: &mut CommandEncoder,
        src: &TextureView,
        dst: &TextureView,
        colors: &Buffer,
    ) {
        let pipeline = &self.starburst_filter_pipeline;
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

        render_pass.set_pipeline(pipeline);

        render_pass.set_bind_group(0, &texture_bind_group, &[]);
        render_pass.set_bind_group(1, &colors_bind_group, &[]);

        render_pass.draw(0..3, 0..1);
    }
}