mod arc;
mod bind_group;
mod buffer;
mod camera;
pub mod lenses;
mod mesh;
mod render_pipeline_descriptor;
mod software;
mod uniforms;
mod vertex;

use crate::arc::Arc;
use crate::bind_group::BindGroup;
use crate::buffer::{Buffer, BufferType};
use crate::camera::{Camera, CameraController};
use crate::lenses::{parse_lenses, LensInterface};
use crate::mesh::Mesh;
use crate::software::{trace, Ray};
use crate::uniforms::{BouncesAndLengthsUniform, CameraUniform, ParamsUniform};
use crate::vertex::{ColoredVertex, Vertex};
use anyhow::*;
use cgmath::Zero;
use color::{AlphaColor, ColorSpace, Hsl, LinearSrgb, Rgba8, Srgb};
use encase::StorageBuffer;
use glam::{vec3, vec4, Vec2, Vec3, Vec3Swizzles, Vec4};
use imgui::{Condition, FontSource, MouseCursor};
use imgui_wgpu::{Renderer, RendererConfig};
use imgui_winit_support::WinitPlatform;
use render_pipeline_descriptor::RenderPipelineDescriptor;
use software::trace_iterator::TraceIterator;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::time::{Duration, Instant};
use uniforms::{LensInterfaceUniform, LensSystemUniform};
use wgpu::util::DeviceExt;
use wgpu::{BufferAddress, ShaderStages};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{
    DeviceEvent, DeviceId, ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta,
    WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

struct ImguiState {
    context: imgui::Context,
    platform: WinitPlatform,
    renderer: Renderer,
    last_cursor: Option<MouseCursor>,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
enum PipelineId {
    Fill,
    Wireframe,
    Lines,
}

const GRID_SIZE_LOG2: u32 = 6;
const GRID_SIZE: u32 = 1 << GRID_SIZE_LOG2;
const GRID_SIZE_VERTEX_COUNT: u32 = GRID_SIZE + 1;
const GRID_VERTEX_BUFFER_SIZE: u32 = GRID_SIZE_VERTEX_COUNT * GRID_SIZE_VERTEX_COUNT;
const GRID_INDEX_BUFFER_SIZE: u32 = GRID_SIZE * GRID_SIZE * 6;

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    window: std::sync::Arc<Window>,
    render_pipelines: HashMap<PipelineId, wgpu::RenderPipeline>,
    // depth_texture: texture::Texture,
    camera: Camera,
    projection: camera::Projection,
    camera_controller: CameraController,
    camera_bind_group: BindGroup<CameraUniform>,

    // inputs_bind_group: BindGroup<InputsUniform>,
    // inputs_bind_group2: BindGroup<InputsUniform>,
    mouse_left_pressed: bool,
    mouse_right_pressed: bool,
    imgui: ImguiState,
    last_frame: Instant,

    lenses_bind_group: BindGroup<LensSystemUniform>,
    bounces_and_lengths_uniform: Box<BouncesAndLengthsUniform>,
    bounces_and_lengths_bind_group: wgpu::BindGroup,
    params_bind_group: wgpu::BindGroup,
    params_uniform: Buffer<ParamsUniform>,
    grid_mesh: Mesh,
    static_lines_vertices: Vec<ColoredVertex>,
    static_lines_vertex_buffer: wgpu::Buffer,
    ray_lines_vertices: Vec<ColoredVertex>,
    ray_lines_vertex_buffer: wgpu::Buffer,
    selected_lens: isize,
    selected_ray: isize,
    ray_step_count: usize,
    debug_mode: bool,
}

const DEBUG_RAY_COUNT: isize = 48 + 1;

impl State {
    pub async fn new(window: std::sync::Arc<Window>) -> Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::POLYGON_MODE_POINT
                    | wgpu::Features::POLYGON_MODE_LINE,
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);

        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo, //surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let projection =
            camera::Projection::new(config.width, config.height, cgmath::Deg(45.0), 0.1, 100.0);

        let camera = Camera::new(
            (0.0, 0.0, -1.0),  // target at world origin
            cgmath::Deg(90.0), // yaw
            cgmath::Deg(0.0),  // pitch
        );
        let camera_controller = CameraController::new(4.0, 1.75);

        let camera_uniform = Buffer::new(
            &device,
            "Camera Buffer",
            BufferType::Uniform,
            CameraUniform::from_camera_and_projection(&camera, &projection),
        )?;

        let camera_bind_group = BindGroup::new(&device, ShaderStages::VERTEX, camera_uniform);

        let lenses = parse_lenses(include_str!("../lenses/wide.22mm.dat"))
            .map_err(|e| format_err!("Failed to parse lenses: {}", e))?;

        let mut lenses_uniform = LensSystemUniform::from(&lenses);

        lenses_uniform.interfaces[1].d1 = 96.4;

        let bounces_and_lengths_uniform = Box::new(BouncesAndLengthsUniform::new(
            lenses_uniform.interface_count as usize,
            lenses_uniform.aperture_index as usize,
        ));

        let bounces_and_lengths_data: Vec<u8> =
            StorageBuffer::<BouncesAndLengthsUniform>::content_of(&*bounces_and_lengths_uniform)
                .map_err(|e| {
                    format_err!("Failed to create bounces and lengths storage buffer: {}", e)
                })?;

        let bounces_and_lengths_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Bounces and Lengths Uniform Buffer"),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                contents: &bounces_and_lengths_data,
            });

        let params = ParamsUniform {
            light_pos: Vec3::new(-0.5, -0.5, 50.0),
            bid: -1,
            intensity: 0.5,
            lambda: 520.0, // 520 nm is some green color.
            wireframe: 0,
        };

        let params_uniform = Buffer::new(&device, "Tracing Params", BufferType::Uniform, params)?;

        let mut static_lines_vertices = Vec::<ColoredVertex>::new();

        // Draw axes
        {
            for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                static_lines_vertices.push(ColoredVertex::new(axis, axis.extend(0.5)));
                static_lines_vertices.push(ColoredVertex::new(-axis, axis.extend(0.1)));
            }
        }

        let static_lines_vertex_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Lines Vertex Buffer"),
                usage: wgpu::BufferUsages::VERTEX,
                contents: bytemuck::cast_slice(&static_lines_vertices),
            });

        let ray_lines_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lines Vertex Buffer"),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            size: (1 << 16) * size_of::<ColoredVertex>() as BufferAddress,
            mapped_at_creation: false,
        });

        let lenses_uniform_buffer =
            Buffer::new(&device, "Lens System", BufferType::Uniform, lenses_uniform)?;

        let lenses_bind_group =
            BindGroup::new(&device, ShaderStages::VERTEX, lenses_uniform_buffer);

        let bounces_and_lengths_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bounces and Lengths Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let bounces_and_lengths_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bounces and Lengths Bind Group"),
            layout: &bounces_and_lengths_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: bounces_and_lengths_buffer.as_entire_binding(),
            }],
        });

        let params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Params Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Params Bind Group"),
            layout: &params_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_uniform.buffer().as_entire_binding(),
            }],
        });

        let render_pipelines = Self::setup_render_pipelines(
            &device,
            &config,
            &[
                &camera_bind_group.bind_group_layout(),
                &lenses_bind_group.bind_group_layout(),
                &bounces_and_lengths_bind_group_layout,
                &params_bind_group_layout,
            ],
        );

        let grid_vertices = {
            let mut grid_vertices =
                Vec::with_capacity((GRID_SIZE_VERTEX_COUNT * GRID_SIZE_VERTEX_COUNT) as usize);

            for y in 0..GRID_SIZE_VERTEX_COUNT {
                for x in 0..GRID_SIZE_VERTEX_COUNT {
                    grid_vertices.push(Vertex {
                        position: Vec3::new(
                            x as f32 / GRID_SIZE as f32,
                            y as f32 / GRID_SIZE as f32,
                            0.0,
                        ),
                    });
                }
            }

            grid_vertices
        };

        let grid_indices: Vec<u32> = {
            let mut grid_indices = Vec::with_capacity(GRID_INDEX_BUFFER_SIZE as usize);

            for y in 0..GRID_SIZE {
                for x in 0..GRID_SIZE {
                    let tl = y * GRID_SIZE_VERTEX_COUNT + x;
                    grid_indices.push(tl);
                    grid_indices.push(tl + 1);
                    grid_indices.push(tl + GRID_SIZE_VERTEX_COUNT + 0);
                    grid_indices.push(tl + 1);
                    grid_indices.push(tl + GRID_SIZE_VERTEX_COUNT + 1);
                    grid_indices.push(tl + GRID_SIZE_VERTEX_COUNT + 0);
                }
            }

            grid_indices
        };

        let grid_mesh = Mesh::new(&device, &grid_vertices, &grid_indices);

        let imgui = Self::setup_imgui(&window, &device, &queue, surface_format);

        {
            // let r_pos = Vec3::new(0.0, 0.0, 0.0);
            //
            // for bid in 0..35 {
            //     println!("bid {bid}");
            //
            //     for vert in &grid_vertices {
            //         let offset = vert.position - Vec3::new(-0.5, -0.5, 0.0);
            //         let offset = offset * lenses[0].aperture;
            //
            //         let lens_entrance_center = Vec3::new(0.0, 0.0, lenses[0].axis_position);
            //
            //         // Set lens entrance center to the center of the lens with an offset relative to the ray's position and the lens' radius,
            //         // i.e., radius * 0.5 away from center of the lens.
            //         let lens_hit = lens_entrance_center + offset;
            //
            //         let r_dir = (lens_hit - r_pos).normalize();
            //         let r = Ray {
            //             pos: r_pos,
            //             dir: r_dir,
            //             tex: Vec4::new(0.5, 0.5, 0.5, 1.0),
            //         };
            //
            //         let lenses = LensSystemUniform::from(&lenses);
            //         let lengths_and_bounces = LengthsAndBouncesUniform::new(lenses.interface_count as usize);
            //
            //         trace(bid, &r, 500.0, &lenses.interfaces, &lengths_and_bounces.data, lenses.aperture_index as usize);
            //     }
            // }
        }

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            window,
            render_pipelines,
            // depth_texture,
            camera,
            projection,
            camera_controller,
            camera_bind_group,
            // inputs_bind_group,
            // inputs_bind_group2,
            lenses_bind_group,
            bounces_and_lengths_uniform,
            bounces_and_lengths_bind_group,
            params_bind_group,
            params_uniform,
            grid_mesh,
            static_lines_vertices,
            static_lines_vertex_buffer,
            ray_lines_vertices: Vec::with_capacity(1 << 18),
            ray_lines_vertex_buffer,
            selected_lens: -1,
            selected_ray: -1,
            ray_step_count: 30,

            mouse_left_pressed: false,
            mouse_right_pressed: false,
            imgui,
            last_frame: Instant::now(),

            debug_mode: false,
        })
    }

    fn build_lens_system_debug_lines(
        lenses_uniform: &LensSystemUniform,
        bounces_and_lengths: &BouncesAndLengthsUniform,
        line_vertices: &mut Vec<ColoredVertex>,
        selected_lens: Option<usize>,
        selected_bid: Option<usize>,
    ) {
        let lens_range = match selected_lens {
            Some(lens) => lens..lens + 1,
            None => 0..lenses_uniform.interface_count as usize,
        };

        // for i in [2u32, 5] {
        // for i in 0..3 {
        for i in lens_range {
            let lens = &lenses_uniform.interfaces[i];
            let bounces = selected_bid.map(|bid| bounces_and_lengths.data[bid].xy());

            let lens_center = lens.center;

            if lens.radius == 0.0 {
                let col = Vec4::new(1.0, 1.0, 0.0, 0.3);

                line_vertices.push(ColoredVertex::new(
                    lens_center + Vec3::Y * lens.sa * 0.5,
                    col,
                ));
                line_vertices.push(ColoredVertex::new(
                    lens_center + Vec3::Y * (lens.sa * 0.5 + 0.02),
                    col,
                ));
                line_vertices.push(ColoredVertex::new(
                    lens_center - Vec3::Y * lens.sa * 0.5,
                    col,
                ));
                line_vertices.push(ColoredVertex::new(
                    lens_center - Vec3::Y * (lens.sa * 0.5 + 0.02),
                    col,
                ));

                // println!("{i:2}\t{:?}\t{:?}", lens_center, lens.radius);
            } else if i == 0 {
                // let col = Vec4::new(0.0, 1.0, 1.0, 0.3);
                //
                // line_vertices.push(ColoredVertex::new(lens.center + Vec3::Y * lens.sa, col));
                // line_vertices.push(ColoredVertex::new(lens.center - Vec3::Y * lens.sa, col));
            } else if lens.flat_surface == 1 {
                let col = Vec4::new(1.0, 0.0, 1.0, 0.3);

                line_vertices.push(ColoredVertex::new(
                    lens_center + Vec3::Y * lens.sa * 0.5,
                    col,
                ));
                line_vertices.push(ColoredVertex::new(
                    lens_center - Vec3::Y * lens.sa * 0.5,
                    col,
                ));

                // println!("{i:2}\t{:?}\t{:?}", lens.center, lens.radius);
            } else {
                let col = match i {
                    x if bounces.is_some_and(|v| x as u32 == v.x) => vec4(0.0, 1.0, 0.0, 0.3),
                    x if bounces.is_some_and(|v| x as u32 == v.y) => vec4(0.0, 0.0, 1.0, 0.3),
                    _ => vec4(
                        1.0,
                        0.5 * (i as f32 / lenses_uniform.interface_count as f32),
                        0.0,
                        1.0,
                    ),
                };

                let angle = (lens.sa * 0.5).atan2(lens.radius);
                let arc_center = lens_center.zy();

                let (start_angle, end_angle) = if lens.radius > 0.0 {
                    (-angle, angle)
                } else {
                    (angle - PI, PI - angle)
                };

                // println!("{i:2}\t{:?}\t{:?}\t{:?}", lens_center, lens.radius, arc_center);

                let arc_verts = Arc::new(arc_center, lens.radius, start_angle, end_angle)
                    .iter(16)
                    .collect::<Vec<_>>();

                let arc_verts = arc_verts[..]
                    .windows(2)
                    .flat_map(|a| {
                        [
                            ColoredVertex::new(Vec3::new(0.0, a[0].y, a[0].x), col),
                            ColoredVertex::new(Vec3::new(0.0, a[1].y, a[1].x), col),
                        ]
                    })
                    .collect::<Vec<_>>();

                let col = Vec3::ONE.extend(0.05);

                let arc_center = Vec3::new(0.0, arc_center.y, arc_center.x);

                line_vertices.push(ColoredVertex::new(arc_center, col));
                line_vertices.push(ColoredVertex::new(arc_verts[0].position.into(), col));
                line_vertices.push(ColoredVertex::new(arc_center, col));
                line_vertices.push(ColoredVertex::new(
                    arc_verts[arc_verts.len() - 1].position.into(),
                    col,
                ));

                line_vertices.extend(arc_verts);
            }
        }
    }

    fn build_ray_traces_debug_lines(
        lenses_uniform: &LensSystemUniform,
        bounces_and_lengths: &BouncesAndLengthsUniform,
        line_vertices: &mut Vec<ColoredVertex>,
        ray_count: usize,
        selected_ray: Option<usize>,
        selected_bid: Option<usize>,
        ray_intensity: f32,
        ray_lambda: f32,
        ray_step_count: usize,
    ) {
        let ray_range = match selected_ray {
            Some(ray) => ray..ray + 1,
            None => 0..ray_count,
        };

        for i in ray_range {
            let entrance = &lenses_uniform.interfaces[0];

            let t = i as f32 / (ray_count - 1) as f32;

            let ray = Ray {
                pos: Vec3::new(0.0, entrance.center.y - (t - 0.5) * entrance.sa, 0.5),
                dir: Vec3::new(0.0, 0.0, -1.0),
                tex: Vec4::new(0.5, t, entrance.sa, ray_intensity),
            };

            let iter = TraceIterator::new(
                &lenses_uniform.interfaces,
                &bounces_and_lengths.data,
                lenses_uniform.aperture_index as usize,
                ray,
                selected_bid.unwrap_or(0),
                ray_lambda,
            );

            // let color = Hsl::to_linear_srgb([t * 360., 75., 75.]);
            // let color = Vec3::from_array(color).extend(0.3);

            line_vertices.extend(iter.take(ray_step_count).flat_map(|ray| {
                let color = if ray.intensity.is_zero() {
                    vec4(1.0, 0.0, 0.0, 0.05)
                } else {
                    Vec3::ONE.extend(0.2)
                };

                [
                    ColoredVertex::new(ray.start, color),
                    ColoredVertex::new(ray.end, color),
                ]
            }));
        }
    }

    fn setup_render_pipelines(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        bind_group_layouts: &[&wgpu::BindGroupLayout],
    ) -> HashMap<PipelineId, wgpu::RenderPipeline> {
        let mut render_pipelines = HashMap::new();

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let lines_shader = device.create_shader_module(wgpu::include_wgsl!("lines.wgsl"));

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts,
                push_constant_ranges: &[],
            });

        let render_pipeline_default =
            RenderPipelineDescriptor::new(render_pipeline_layout.clone(), shader, config.format);

        let render_pipeline_default_desc = render_pipeline_default.desc();

        let mut render_pipeline_wireframe_desc = render_pipeline_default_desc.clone();
        render_pipeline_wireframe_desc.label = Some("Render Wireframe Pipeline");
        render_pipeline_wireframe_desc.primitive.polygon_mode = wgpu::PolygonMode::Line;
        render_pipeline_wireframe_desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;

        render_pipelines.insert(
            PipelineId::Fill,
            device.create_render_pipeline(&render_pipeline_default_desc),
        );
        render_pipelines.insert(
            PipelineId::Wireframe,
            device.create_render_pipeline(&render_pipeline_wireframe_desc),
        );

        {
            let render_pipeline_lines_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Lines Pipeline Layout"),
                    bind_group_layouts: &[bind_group_layouts[0]],
                    push_constant_ranges: &[],
                });

            let targets = [Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(render_pipeline_descriptor::ADDITIVE_BLEND),
                write_mask: wgpu::ColorWrites::ALL,
            })];

            let render_pipeline_lines_desc = wgpu::RenderPipelineDescriptor {
                label: Some("Render Lines Pipeline"),
                layout: Some(&render_pipeline_lines_layout),
                vertex: wgpu::VertexState {
                    module: &lines_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[ColoredVertex::desc()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &lines_shader,
                    entry_point: Some("fs_main"),
                    targets: &targets,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Cw,
                    cull_mode: None, // Some(wgpu::Face::Back),
                    // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                    polygon_mode: wgpu::PolygonMode::Line,
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
            };

            render_pipelines.insert(
                PipelineId::Lines,
                device.create_render_pipeline(&render_pipeline_lines_desc),
            );
        }
        // render_pipelines.insert(
        //     PipelineId::Points,
        //     device.create_render_pipeline(&render_pipeline_instance_quads_descriptor),
        // );

        render_pipelines
    }

    fn setup_imgui(
        window: &std::sync::Arc<Window>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
    ) -> ImguiState {
        let mut context = imgui::Context::create();
        let mut platform = WinitPlatform::new(&mut context);
        platform.attach_window(
            context.io_mut(),
            &window,
            imgui_winit_support::HiDpiMode::Default,
        );
        context.set_ini_filename(None);

        let hidpi_factor = window.scale_factor();
        let font_size = (13.0 * hidpi_factor) as f32;
        context.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

        context.fonts().add_font(&[FontSource::DefaultFontData {
            config: Some(imgui::FontConfig {
                oversample_h: 1,
                pixel_snap_h: true,
                size_pixels: font_size,
                ..Default::default()
            }),
        }]);

        let renderer_config = RendererConfig {
            texture_format: surface_format,
            ..Default::default()
        };

        let renderer = Renderer::new(&mut context, &device, &queue, renderer_config);
        let last_cursor = None;

        ImguiState {
            context,
            platform,
            renderer,
            last_cursor,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            // self.depth_texture =
            //     texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
            // self.projection.resize(width, height);
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
        }
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, key: KeyCode, pressed: bool) {
        if !self.camera_controller.handle_key(key, pressed) {
            match (key, pressed) {
                (KeyCode::Escape, true) => event_loop.exit(),
                _ => {}
            }
        }
    }

    fn handle_mouse_button(&mut self, button: MouseButton, pressed: bool) {
        match button {
            MouseButton::Left => {
                self.mouse_left_pressed = pressed;
            }
            MouseButton::Right => {
                self.mouse_right_pressed = pressed;
            }
            _ => {}
        }

        self.camera_controller.handle_mouse_button(button, pressed);
    }

    fn handle_mouse_scroll(&mut self, delta: &MouseScrollDelta) {
        self.camera_controller.handle_mouse_scroll(delta);
    }

    fn update(&mut self, dt: Duration) {
        self.camera_controller.update_camera(&mut self.camera, dt);

        self.camera_bind_group
            .buffer_mut()
            .data
            .update_view_proj(&self.camera, &self.projection);
        self.camera_bind_group
            .buffer()
            .write_buffer(&self.queue)
            .unwrap();

        self.imgui.context.io_mut().update_delta_time(dt);

        self.params_uniform.write_buffer(&self.queue).unwrap();

        if self.debug_mode {
            self.ray_lines_vertices.clear();

            let selected_bid = if self.params_uniform.data.bid >= 0 {
                Some(self.params_uniform.data.bid as usize)
            } else {
                None
            };

            let selected_lens = if self.selected_lens >= 0 {
                Some(self.selected_lens as usize)
            } else {
                None
            };

            let selected_ray = if self.selected_ray >= 0 {
                Some(self.selected_ray as usize)
            } else {
                None
            };

            Self::build_lens_system_debug_lines(
                &self.lenses_bind_group.buffer().data,
                &self.bounces_and_lengths_uniform,
                &mut self.ray_lines_vertices,
                selected_lens,
                selected_bid,
            );

            Self::build_ray_traces_debug_lines(
                &self.lenses_bind_group.buffer().data,
                &self.bounces_and_lengths_uniform,
                &mut self.ray_lines_vertices,
                DEBUG_RAY_COUNT as usize,
                selected_ray,
                selected_bid,
                1.0,
                520.0,
                self.ray_step_count,
            );

            self.queue.write_buffer(
                &self.ray_lines_vertex_buffer,
                0,
                bytemuck::cast_slice(&self.ray_lines_vertices),
            );
        }
    }

    pub fn render(&mut self) -> Result<()> {
        self.window.request_redraw();

        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Render flare
        if !self.debug_mode {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Additive Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_bind_group(0, self.camera_bind_group.bind_group(), &[]);
            render_pass.set_bind_group(1, self.lenses_bind_group.bind_group(), &[]);
            render_pass.set_bind_group(2, &self.bounces_and_lengths_bind_group, &[]);
            render_pass.set_bind_group(3, &self.params_bind_group, &[]);

            self.grid_mesh.bind(&mut render_pass);

            let pipeline = if self.params_uniform.data.wireframe == 1 {
                PipelineId::Wireframe
            } else {
                PipelineId::Fill
            };

            render_pass.set_pipeline(&self.render_pipelines[&pipeline]);

            let bounce_count = if self.params_uniform.data.bid < 0 {
                self.lenses_bind_group.buffer().data.bounce_count
            } else {
                1
            };

            self.grid_mesh
                .draw_instanced(&mut render_pass, 3 * bounce_count);
        }

        // Render debug
        if self.debug_mode {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Additive Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            render_pass.set_bind_group(0, self.camera_bind_group.bind_group(), &[]);

            render_pass.set_pipeline(&self.render_pipelines[&PipelineId::Lines]);

            render_pass.set_vertex_buffer(0, self.static_lines_vertex_buffer.slice(..));
            render_pass.draw(0..self.static_lines_vertices.len() as u32, 0..1);

            render_pass.set_vertex_buffer(0, self.ray_lines_vertex_buffer.slice(..));
            render_pass.draw(0..self.ray_lines_vertices.len() as u32, 0..1);
        }

        // Render UI
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("UI Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            self.render_imgui(&mut render_pass)?;
        }

        self.queue.submit(Some(encoder.finish()));

        output.present();

        Ok(())
    }

    fn render_imgui<'a>(&'a mut self, render_pass: &mut wgpu::RenderPass<'a>) -> Result<()> {
        let imgui = &mut self.imgui;

        imgui
            .platform
            .prepare_frame(imgui.context.io_mut(), &self.window)
            .expect("Failed to prepare frame");
        let ui = imgui.context.frame();

        {
            let window = ui.window("Debug");
            window
                .size([200.0, 40.0], Condition::FirstUseEver)
                .position([10.0, 10.0], Condition::FirstUseEver)
                .no_decoration()
                .no_nav()
                .always_auto_resize(true)
                .focus_on_appearing(false)
                .save_settings(false)
                .build(|| {
                    ui.text(format!("Frametime: {:.2?}", self.last_frame.elapsed()));
                    ui.text(format!("Mouse: {:?}", ui.io().mouse_pos));
                    ui.text(format!(
                        "Camera Yaw: {:.1?}",
                        self.camera.yaw.0.to_degrees()
                    ));
                    ui.text(format!(
                        "Camera Pitch: {:.1?}",
                        self.camera.pitch.0.to_degrees()
                    ));
                    // ui.text(format!(
                    //     "Mouse Drag Pos: {:?}",
                    //     self.camera_controller.mouse_position
                    // ));

                    // let screen_size =
                    //     Point2::new(self.config.width as f64, self.config.height as f64);
                    // let mouse_point = remap_neg1_1(
                    //     self.camera_controller.mouse_position.to_vec(),
                    //     Vector2::zero(),
                    //     screen_size.to_vec(),
                    // );
                    // // let mouse_point = point_on_unit_sphere(mouse_point);
                    // ui.text(format!("Mouse Normalized: {:.1?}", mouse_point));

                    ui.separator();
                    ui.group(|| {
                        ui.radio_button("Flare", &mut self.debug_mode, false);
                        ui.radio_button("Lens Model", &mut self.debug_mode, true);
                    });

                    ui.separator();
                    ui.slider(
                        "Render Wireframe",
                        0,
                        1,
                        &mut self.params_uniform.data.wireframe,
                    );

                    ui.text(format!("Ray Pos: {:?}", self.params_uniform.data.light_pos));

                    let bid = self.params_uniform.data.bid.max(0);
                    let bounces_and_length = self.bounces_and_lengths_uniform.data[bid as usize];
                    ui.text(format!(
                        "{}->{} ({})",
                        bounces_and_length.x, bounces_and_length.y, bounces_and_length.z
                    ));

                    let max_bounces = self.lenses_bind_group.buffer().data.bounce_count as i32;

                    ui.slider(
                        "Bounce ID",
                        -1,
                        max_bounces,
                        &mut self.params_uniform.data.bid,
                    );
                    ui.slider(
                        "Ray Intensity",
                        0.0,
                        5.0,
                        &mut self.params_uniform.data.intensity,
                    );
                    ui.slider(
                        "Wavelength",
                        360.0,
                        830.0,
                        &mut self.params_uniform.data.lambda,
                    );
                    ui.slider(
                        "Selected Lens",
                        -1,
                        self.lenses_bind_group.buffer().data.interface_count as isize,
                        &mut self.selected_lens,
                    );
                    ui.slider("Select Ray", -1, DEBUG_RAY_COUNT, &mut self.selected_ray);
                    ui.slider("Ray Step Count", 0, 100, &mut self.ray_step_count);
                });

            // ui.show_demo_window(&mut imgui.demo_open);
        }

        if imgui.last_cursor != ui.mouse_cursor() {
            imgui.last_cursor = ui.mouse_cursor();
            imgui.platform.prepare_render(ui, &self.window);
        }

        imgui
            .renderer
            .render(
                imgui.context.render(),
                &self.queue,
                &self.device,
                render_pass,
            )
            .expect("Rendering failed");

        Ok(())
    }
}

struct App {
    state: Option<State>,
}

impl App {
    fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes =
            Window::default_attributes().with_inner_size(LogicalSize::new(1920, 1080));

        let window = std::sync::Arc::new(event_loop.create_window(window_attributes).unwrap());

        self.state = Some(pollster::block_on(State::new(window)).unwrap());
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        let imgui = &mut event.imgui;

        imgui.platform.handle_event::<()>(
            imgui.context.io_mut(),
            &event.window,
            &Event::UserEvent(()),
        );

        self.state = Some(event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match &event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now - state.last_frame;

                state.last_frame = now;

                state.update(dt);

                if let Err(err) = state.render() {
                    match err.downcast_ref::<wgpu::SurfaceError>() {
                        Some(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            let size = state.window.inner_size();
                            state.resize(size.width, size.height);
                        }
                        _ => {
                            log::error!("Unable to render: {:?}", err);
                        }
                    }
                }
            }
            WindowEvent::MouseInput {
                state: btn_state,
                button,
                ..
            } => {
                if !state.imgui.context.io().want_capture_mouse {
                    state.handle_mouse_button(*button, btn_state.is_pressed())
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if !state.imgui.context.io().want_capture_mouse {
                    state.handle_mouse_scroll(&delta);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Mouse cursor position handling - not needed for camera control
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => {
                if !state.imgui.context.io().want_capture_keyboard {
                    state.handle_key(event_loop, *code, key_state.is_pressed())
                }
            }
            _ => {}
        }

        let imgui = &mut state.imgui;

        imgui.platform.handle_event::<State>(
            imgui.context.io_mut(),
            &state.window,
            &Event::WindowEvent { window_id, event },
        );
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let state = if let Some(state) = &mut self.state {
            state
        } else {
            return;
        };

        match event {
            DeviceEvent::MouseMotion { delta } => {
                if !state.imgui.context.io().want_capture_mouse {
                    if state.mouse_left_pressed {
                        state.camera_controller.handle_mouse(delta.0, delta.1);
                    }

                    if state.mouse_right_pressed {
                        let delta = Vec2::new(-delta.0 as f32, -delta.1 as f32).extend(0.0) * 0.25;

                        state.params_uniform.data.light_pos += delta;
                    }
                }
            }
            _ => {}
        }

        let imgui = &mut state.imgui;

        imgui.platform.handle_event::<State>(
            imgui.context.io_mut(),
            &state.window,
            &Event::DeviceEvent { device_id, event },
        );
    }
}

pub fn run() -> Result<()> {
    env_logger::init();

    let event_loop = EventLoop::with_user_event().build()?;

    let mut app = App::new();

    event_loop.run_app(&mut app)?;

    Ok(())
}

pub fn remap(v: f32, from_min: f32, from_max: f32, to_min: f32, to_max: f32) -> f32 {
    let from_range = from_max - from_min;
    let to_range = to_max - to_min;

    (v - from_min) * (to_range / from_range) + to_min
}
