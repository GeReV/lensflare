use crate::shaders::create_shader;
use crate::vertex::Vertex;
use glam::Vec3;
use std::f32::consts::TAU;
use wesl::{StandardResolver, Wesl};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::wgt::{TextureDescriptor, TextureViewDescriptor};
use wgpu::{
    BlendState, Color, ColorTargetState, ColorWrites, CommandEncoderDescriptor, Device, Extent3d,
    FragmentState, FrontFace, IndexFormat, LoadOp, MultisampleState, Operations,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PolygonMode, PrimitiveState,
    PrimitiveTopology, Queue, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipelineDescriptor, StoreOp, Texture, TextureDimension, TextureFormat, TextureUsages,
    VertexState,
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
        format: TextureFormat::Rgba8Unorm,
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
            format: TextureFormat::Rgba8Unorm,
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
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Ghost Index Buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
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
    start += if sides % 2 == 0 {
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
