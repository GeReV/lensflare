use wgpu::{BlendComponent, BlendFactor, BlendOperation, BlendState};
use crate::vertex::Vertex;

pub struct RenderPipelineDescriptor<'a> {
    layout: wgpu::PipelineLayout,
    shader: wgpu::ShaderModule,
    fragment_targets: Box<[Option<wgpu::ColorTargetState>]>,
    vertex_buffer_layout: Box<[wgpu::VertexBufferLayout<'a>]>,
}

pub const ADDITIVE_BLEND: BlendState = BlendState {
    color: BlendComponent {
        src_factor: BlendFactor::SrcAlpha,
        dst_factor: BlendFactor::OneMinusSrcAlpha,
        operation: BlendOperation::Add,
    },
    alpha: BlendComponent {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::One,
        operation: BlendOperation::Add,
    },
};

impl<'a> RenderPipelineDescriptor<'a> {
    pub fn new(
        layout: wgpu::PipelineLayout,
        shader: wgpu::ShaderModule,
        format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            layout,
            shader,
            fragment_targets: Box::new([Some(wgpu::ColorTargetState {
                format,
                blend: Some(ADDITIVE_BLEND),
                write_mask: wgpu::ColorWrites::ALL,
            })]),
            vertex_buffer_layout: Box::new([Vertex::desc()]),
        }
    }

    pub fn desc(&self) -> wgpu::RenderPipelineDescriptor {
        wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&self.layout),
            vertex: wgpu::VertexState {
                module: &self.shader,
                entry_point: Some("vs_main"),
                buffers: &self.vertex_buffer_layout,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: Some("fs_main"),
                targets: &self.fragment_targets,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: None,// Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
                // Some(wgpu::DepthStencilState {
                //     format: texture::Texture::DEPTH_FORMAT,
                //     depth_write_enabled: true,
                //     depth_compare: wgpu::CompareFunction::Less,
                //     stencil: wgpu::StencilState::default(),
                //     bias: wgpu::DepthBiasState::default(),
                // }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        }
    }
}