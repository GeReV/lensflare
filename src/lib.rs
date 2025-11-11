mod arc;
mod buffer;
mod camera;
pub mod lenses;
mod mesh;
mod registry;
mod software;
mod uniforms;
mod vertex;
mod hot_reload;

use crate::arc::Arc;
use crate::camera::{Camera, CameraController, Projection};
use crate::lenses::{parse_lenses};
use crate::mesh::Mesh;
use crate::registry::{Id, Registry};
use crate::software::{Ray};
use crate::uniforms::{
    BouncesAndLengthsUniform, CameraUniform, GridLimitsUniform, ParamsUniform, Uniform,
};
use crate::vertex::{ColoredVertex, Vertex};
use anyhow::*;
use cgmath::Zero;
use encase::{StorageBuffer, UniformBuffer};
use glam::{vec3, vec4, Vec2, Vec3, Vec3Swizzles, Vec4};
use imgui::{Condition, FontSource, MouseCursor};
use imgui_wgpu::{Renderer, RendererConfig};
use imgui_winit_support::WinitPlatform;
use itertools::Itertools;
use software::trace_iterator::TraceIterator;
use std::borrow::Cow;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::ffi::OsStr;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime};
use uniforms::{LensSystemUniform};
use wgpu::util::DeviceExt;
use wgpu::wgt::{TextureViewDescriptor};
use wgpu::{
    BindGroupLayout, BlendComponent, BlendState, BufferAddress, BufferUsages, Device, Extent3d,
    Label, Origin3d, PrimitiveState, PrimitiveTopology, ShaderModule, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, SurfaceConfiguration, TextureAspect, TextureFormat,
    TextureSampleType,
};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{
    DeviceEvent, DeviceId, ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta,
    WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};
use crate::hot_reload::{HotReloadResult, HotReloadShader};

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
    Fullscreen,
}

const GRID_SIZE_LOG2: u32 = 6;
const GRID_SIZE: u32 = 1 << GRID_SIZE_LOG2;
const GRID_SIZE_VERTEX_COUNT: u32 = GRID_SIZE + 1;
const GRID_VERTEX_BUFFER_SIZE: u32 = GRID_SIZE_VERTEX_COUNT * GRID_SIZE_VERTEX_COUNT;
const GRID_INDEX_BUFFER_SIZE: u32 = GRID_SIZE * GRID_SIZE * 6;

const ADDITIVE_BLEND: BlendState = BlendState {
    color: BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    window: std::sync::Arc<Window>,
    render_pipelines: HashMap<PipelineId, wgpu::RenderPipeline>,

    render_target: wgpu::Texture,

    camera: Camera,
    projection: camera::Projection,
    camera_controller: CameraController,

    hot_reload_shaders: Vec<HotReloadShader>,

    // inputs_bind_group: BindGroup<InputsUniform>,
    // inputs_bind_group2: BindGroup<InputsUniform>,
    mouse_left_pressed: bool,
    mouse_right_pressed: bool,
    imgui: ImguiState,
    last_frame: Instant,

    buffers: Registry<wgpu::Buffer>,
    bind_group_layouts: Registry<wgpu::BindGroupLayout>,
    bind_groups: Registry<wgpu::BindGroup>,

    default_shader_bind_group_layouts: Vec<Id<BindGroupLayout>>,

    camera_uniform: Uniform<CameraUniform>,
    lenses_uniform: Uniform<LensSystemUniform>,
    bounces_and_lengths_uniform: Uniform<BouncesAndLengthsUniform>,
    grid_limits_uniform: Uniform<GridLimitsUniform>,
    params_uniform: Uniform<ParamsUniform>,

    fullscreen_bind_group_layout_id: Id<BindGroupLayout>,
    fullscreen_bind_group_id: Id<wgpu::BindGroup>,

    light_angles: Vec4,

    grid_mesh: Mesh,
    static_lines_vertices: Vec<ColoredVertex>,
    static_lines_vertex_buffer: wgpu::Buffer,
    ray_lines_vertices: Vec<ColoredVertex>,
    ray_lines_vertex_buffer: wgpu::Buffer,
    selected_lens: isize,
    selected_ray: isize,
    ray_step_count: usize,

    debug_mode: bool,
    draw_axes: bool,
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

        let mut buffers: Registry<wgpu::Buffer> = Registry::new();
        let mut bind_group_layouts: Registry<wgpu::BindGroupLayout> = Registry::new();
        let mut bind_groups: Registry<wgpu::BindGroup> = Registry::new();

        let size = wgpu::Extent3d {
            width: config.width.max(1),
            height: config.height.max(1),
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("Render Target Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        let render_target = device.create_texture(&desc);

        let (projection, camera, camera_controller, camera_uniform) = Self::create_camera(
            &device,
            &config,
            &mut buffers,
            &mut bind_group_layouts,
            &mut bind_groups,
        )?;

        let lenses = parse_lenses(include_str!("../lenses/wide.22mm.dat"))
            .map_err(|e| format_err!("Failed to parse lenses: {}", e))?;

        let mut lenses_uniform = LensSystemUniform::from(&lenses);

        lenses_uniform.interfaces[1].d1 = 96.4;

        let lenses_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lens System Uniform Buffer"),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            contents: &UniformBuffer::<CameraUniform>::content_of::<_, Vec<u8>>(&lenses_uniform)?,
        });

        let bounces_and_lengths_uniform = BouncesAndLengthsUniform::new(
            lenses_uniform.interface_count as usize,
            lenses_uniform.aperture_index as usize,
        );

        let bounces_and_lengths_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Bounces and Lengths Uniform Buffer"),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                contents: &StorageBuffer::<BouncesAndLengthsUniform>::content_of::<_, Vec<u8>>(
                    &bounces_and_lengths_uniform,
                )?,
            });

        let ray_dir = -Vec3::Z;

        let grid_limits_uniform =
            GridLimitsUniform::new(&lenses_uniform, &bounces_and_lengths_uniform, ray_dir);

        let grid_limits_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Limits Uniform Buffer"),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            contents: &StorageBuffer::<GridLimitsUniform>::content_of::<_, Vec<u8>>(
                &grid_limits_uniform,
            )?,
        });

        let lenses_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lenses Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let lenses_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lenses Bind Group"),
            layout: &lenses_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lenses_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bounces_and_lengths_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grid_limits_buffer.as_entire_binding(),
                },
            ],
        });

        let lenses_bind_group_layout_id = bind_group_layouts.add(lenses_bind_group_layout);
        let lenses_bind_group_id = bind_groups.add(lenses_bind_group);

        let lenses_uniform = Uniform::new(
            lenses_uniform,
            buffers.add(lenses_uniform_buffer),
            lenses_bind_group_layout_id.clone(),
            lenses_bind_group_id.clone(),
        );

        let bounces_and_lengths_uniform = Uniform::new(
            bounces_and_lengths_uniform,
            buffers.add(bounces_and_lengths_buffer),
            lenses_bind_group_layout_id.clone(),
            lenses_bind_group_id.clone(),
        );

        let grid_limits_uniform = Uniform::new(
            grid_limits_uniform,
            buffers.add(grid_limits_buffer),
            lenses_bind_group_layout_id.clone(),
            lenses_bind_group_id.clone(),
        );

        let params = ParamsUniform {
            ray_dir,
            bid: 45,
            intensity: 0.5,
            lambda: 520.0, // 520 nm is some green color.
            wireframe: 0,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tracing Params"),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            contents: &UniformBuffer::<ParamsUniform>::content_of::<_, Vec<u8>>(&params)?,
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
                resource: params_buffer.as_entire_binding(),
            }],
        });

        let params_uniform = Uniform::new(
            params,
            buffers.add(params_buffer),
            bind_group_layouts.add(params_bind_group_layout),
            bind_groups.add(params_bind_group),
        );

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

        let default_shader_bind_group_layouts = vec![
            camera_uniform.bind_group_layout_id.clone(),
            lenses_uniform.bind_group_layout_id.clone(),
            params_uniform.bind_group_layout_id.clone(),
        ];

        let mut render_pipelines = HashMap::new();

        Self::create_static_render_pipelines(
            &device,
            &mut render_pipelines,
            &default_shader_bind_group_layouts
                .iter()
                .map(|bind_group_layout_id| &bind_group_layouts[bind_group_layout_id])
                .collect_vec(),
        );

        let mut hot_reload_shaders = Vec::new();

        hot_reload_shaders.push(HotReloadShader::new("src/shader.wgsl"));
        hot_reload_shaders.push(HotReloadShader::new("src/fullscreen.wgsl"));

        let (fullscreen_bind_group_layout_id, fullscreen_bind_group_id) = {
            let fullscreen_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Render Fullscreen Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                    ],
                });

            let fullscreen_bind_group_layout_id =
                bind_group_layouts.add(fullscreen_bind_group_layout);

            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Render Fullscreen Texture Sampler"),
                ..Default::default()
            });

            let texture_view = render_target.create_view(&TextureViewDescriptor::default());

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Render Fullscreen Bind Group"),
                layout: &bind_group_layouts[&fullscreen_bind_group_layout_id],
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                ],
            });

            let fullscreen_bind_group_id = bind_groups.add(bind_group);

            (fullscreen_bind_group_layout_id, fullscreen_bind_group_id)
        };

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
            buffers,
            bind_group_layouts,
            bind_groups,

            render_target,

            default_shader_bind_group_layouts,

            hot_reload_shaders,

            camera,
            projection,
            camera_controller,

            camera_uniform,
            lenses_uniform,
            bounces_and_lengths_uniform,
            grid_limits_uniform,
            params_uniform,

            fullscreen_bind_group_layout_id,
            fullscreen_bind_group_id,

            light_angles: vec4(0., -PI * 0.5, 0., 0.),

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
            draw_axes: false,
        })
    }

    fn create_camera(
        device: &Device,
        config: &SurfaceConfiguration,
        buffers: &mut Registry<wgpu::Buffer>,
        bind_group_layouts: &mut Registry<BindGroupLayout>,
        bind_groups: &mut Registry<wgpu::BindGroup>,
    ) -> Result<(Projection, Camera, CameraController, Uniform<CameraUniform>), Error> {
        let projection =
            camera::Projection::new(config.width, config.height, cgmath::Deg(45.0), 0.1, 100.0);

        // For a 45 degree projection, having the same distance as the closest lens' radius should
        // have it exactly cover the "sensor".
        let camera_distance = 0.17996 * 0.5;
        let camera = Camera::new(
            (0.0, 0.0, -camera_distance), // target at world origin
            cgmath::Deg(90.0),            // yaw
            cgmath::Deg(0.0),             // pitch
        );
        let camera_controller = CameraController::new(4.0, 1.75);

        let camera_uniform = CameraUniform::from_camera_and_projection(&camera, &projection);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            contents: &UniformBuffer::<CameraUniform>::content_of::<_, Vec<u8>>(&camera_uniform)?,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let camera_uniform = Uniform::new(
            camera_uniform,
            buffers.add(camera_buffer),
            bind_group_layouts.add(camera_bind_group_layout),
            bind_groups.add(camera_bind_group),
        );
        Ok((projection, camera, camera_controller, camera_uniform))
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
            let bounces = selected_bid.map(|bid| bounces_and_lengths.bounces_and_lengths[bid].xy());

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
        ray_grid_limits: &GridLimitsUniform,
        line_vertices: &mut Vec<ColoredVertex>,
        ray_dir: Vec3,
        ray_count: usize,
        selected_ray: Option<usize>,
        selected_bid: Option<usize>,
        ray_intensity: f32,
        ray_lambda: f32,
        ray_step_count: usize,
    ) {
        let entrance = &lenses_uniform.interfaces[0];

        let ray_range = match selected_ray {
            Some(ray) => ray..ray + 1,
            None => 0..ray_count,
        };

        let selected_bid = selected_bid.unwrap_or(0);

        let grid_limits = ray_grid_limits.limits[selected_bid];

        for i in ray_range {
            let t = i as f32 / (ray_count - 1) as f32;

            let bottom = grid_limits.br.with_x(0.0);
            let top = grid_limits.tl.with_x(0.0);
            let ray = Ray {
                pos: bottom + (top - bottom) * t + vec3(0.0, 0.0, 0.5),
                dir: ray_dir,
                tex: Vec4::new(0.5, t, entrance.sa, ray_intensity),
                hit_sensor: true,
            };

            let iter = TraceIterator::new(
                &lenses_uniform.interfaces,
                &bounces_and_lengths.bounces_and_lengths,
                lenses_uniform.aperture_index as usize,
                ray,
                selected_bid,
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

    fn create_static_render_pipelines(device: &Device, render_pipelines: &mut HashMap<PipelineId, wgpu::RenderPipeline>, default_bind_group_layouts: &Vec<&BindGroupLayout>) {
        {
            let lines_shader = device.create_shader_module(wgpu::include_wgsl!("lines.wgsl"));

            let render_pipeline_lines_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Lines Pipeline Layout"),
                    bind_group_layouts: &[default_bind_group_layouts[0]],
                    push_constant_ranges: &[],
                });

            let targets = [Some(wgpu::ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: Some(BlendState::REPLACE),
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
    }

    fn update_render_pipelines(&mut self) -> Result<()> {
        for shader in self.hot_reload_shaders.iter_mut() {
            let Result::Ok(shader_result) = shader.try_hot_reload(&self.device) else {
                continue;
            };

            let shader_module = match shader_result {
                HotReloadResult::Updated(shader_module) => shader_module,
                HotReloadResult::Unchanged => continue,
            };

            if let Some(path) = shader.path.as_os_str().to_str() {
                match path {
                    "src/shader.wgsl" => {
                        let default_bind_group_layouts = self.default_shader_bind_group_layouts.iter()
                            .map(|id| &self.bind_group_layouts[id])
                            .collect_vec();

                        let render_pipeline_default = Self::create_default_render_pipeline(
                            &self.device,
                            None,
                            TextureFormat::Rgba16Float,
                            &shader_module,
                            &default_bind_group_layouts,
                            None,
                        )?;

                        self.render_pipelines.insert(PipelineId::Fill, render_pipeline_default);

                        let render_pipeline_wireframe = Self::create_default_render_pipeline(
                            &self.device,
                            Some("Render Wireframe Pipeline"),
                            TextureFormat::Rgba16Float,
                            &shader_module,
                            &default_bind_group_layouts,
                            Some(PrimitiveState {
                                topology: PrimitiveTopology::LineList,
                                front_face: wgpu::FrontFace::Cw,
                                cull_mode: None,
                                polygon_mode: wgpu::PolygonMode::Line,
                                ..Default::default()
                            }),
                        )?;

                        self.render_pipelines.insert(PipelineId::Wireframe, render_pipeline_wireframe);
                    }
                    "src/fullscreen.wgsl" => {
                        let render_pipeline_layout =
                            self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                                label: Some("Render Fullscreen Pipeline Layout"),
                                bind_group_layouts: &[&self.bind_group_layouts[&self.fullscreen_bind_group_layout_id]],
                                push_constant_ranges: &[],
                            });

                        let render_pipeline_fullscreen =
                            self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                                label: Some("Render Fullscreen Pipeline"),
                                layout: Some(&render_pipeline_layout),
                                vertex: wgpu::VertexState {
                                    module: &shader_module,
                                    entry_point: Some("vs_main"),
                                    buffers: &[],
                                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                                },
                                fragment: Some(wgpu::FragmentState {
                                    module: &shader_module,
                                    entry_point: Some("fs_main"),
                                    targets: &[Some(wgpu::ColorTargetState {
                                        format: self.config.format,
                                        blend: Some(BlendState::REPLACE),
                                        write_mask: wgpu::ColorWrites::ALL,
                                    })],
                                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                                }),
                                primitive: PrimitiveState {
                                    topology: PrimitiveTopology::TriangleList,
                                    strip_index_format: None,
                                    front_face: wgpu::FrontFace::Cw,
                                    cull_mode: Some(wgpu::Face::Back),
                                    // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                                    polygon_mode: wgpu::PolygonMode::Fill,
                                    // Requires Features::DEPTH_CLIP_CONTROL
                                    unclipped_depth: false,
                                    // Requires Features::CONSERVATIVE_RASTERIZATION
                                    conservative: false,
                                },
                                depth_stencil: None,
                                multisample: wgpu::MultisampleState {
                                    count: 1,
                                    mask: !0,
                                    alpha_to_coverage_enabled: false,
                                },
                                multiview: None,
                                cache: None,
                            });

                        self.render_pipelines.insert(PipelineId::Fullscreen, render_pipeline_fullscreen);
                    },
                    _ => {}
                }
            }
        }

        Ok(())
    }

    fn create_default_render_pipeline<'shader>(
        device: &Device,
        label: Label<'shader>,
        texture_format: TextureFormat,
        shader: &ShaderModule,
        bind_group_layouts: &[&BindGroupLayout],
        primitive: Option<PrimitiveState>,
    ) -> Result<wgpu::RenderPipeline> {
        let label = label.unwrap_or("Default Render Pipeline");

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{label} Layout")),
                bind_group_layouts: &bind_group_layouts,
                push_constant_ranges: &[],
            });

        let render_pipeline_default = wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: texture_format,
                    blend: Some(ADDITIVE_BLEND),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: primitive.unwrap_or(PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: None, // Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            }),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        };

        Ok(device.create_render_pipeline(&render_pipeline_default))
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
            let velocity = if pressed { 1.0 } else { 0.0 };
            match (key, pressed) {
                (KeyCode::Numpad4, _) => self.light_angles.w = velocity,
                (KeyCode::Numpad6, _) => self.light_angles.w = -velocity,
                (KeyCode::Numpad8, _) => self.light_angles.z = velocity,
                (KeyCode::Numpad2, _) => self.light_angles.z = -velocity,
                (KeyCode::Escape, true) => event_loop.exit(),
                (KeyCode::Home, true) => self.camera.reset(),
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

        self.camera_uniform
            .data
            .update_view_proj(&self.camera, &self.projection);
        self.camera_uniform
            .write_buffer(&self.queue, &self.buffers)
            .unwrap();

        self.imgui.context.io_mut().update_delta_time(dt);

        self.light_angles.x += dt.as_secs_f32() * self.light_angles.z;
        self.light_angles.y += dt.as_secs_f32() * self.light_angles.w;

        let (sin_pitch, cos_pitch) = self.light_angles.x.sin_cos();
        let (sin_yaw, cos_yaw) = self.light_angles.y.sin_cos();

        let ray_pos = vec3(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize();

        self.params_uniform.data.ray_dir = -ray_pos;

        self.params_uniform
            .write_buffer(&self.queue, &self.buffers)
            .unwrap();

        self.update_render_pipelines().unwrap();

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
                &self.lenses_uniform.data,
                &self.bounces_and_lengths_uniform.data,
                &mut self.ray_lines_vertices,
                selected_lens,
                selected_bid,
            );

            let lens_center = self.lenses_uniform.data.interfaces[0].center;

            let ray_dir = (lens_center - self.params_uniform.data.ray_dir).normalize();

            Self::build_ray_traces_debug_lines(
                &self.lenses_uniform.data,
                &self.bounces_and_lengths_uniform.data,
                &self.grid_limits_uniform.data,
                &mut self.ray_lines_vertices,
                ray_dir,
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

        let view = self
            .render_target
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let output = self.surface.get_current_texture()?;

        // Render flare
        if !self.debug_mode {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Lens Flare Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let binds = [
                &self.camera_uniform.bind_group_id,
                &self.lenses_uniform.bind_group_id,
                &self.params_uniform.bind_group_id,
            ];

            for (i, &bind_group_id) in binds.iter().enumerate() {
                render_pass.set_bind_group(i as u32, &self.bind_groups[bind_group_id], &[]);
            }

            self.grid_mesh.bind(&mut render_pass);

            let pipeline = if self.params_uniform.data.wireframe == 1 {
                PipelineId::Wireframe
            } else {
                PipelineId::Fill
            };

            render_pass.set_pipeline(&self.render_pipelines[&pipeline]);

            let bounce_count = if self.params_uniform.data.bid < 0 {
                self.lenses_uniform.data.bounce_count
            } else {
                1
            };

            self.grid_mesh
                .draw_instanced(&mut render_pass, 3 * bounce_count);
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Lines Pass"),
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

            render_pass.set_bind_group(
                0,
                &self.bind_groups[&self.camera_uniform.bind_group_id],
                &[],
            );

            render_pass.set_pipeline(&self.render_pipelines[&PipelineId::Lines]);

            self.ray_lines_vertices.clear();
            self.ray_lines_vertices
                .push(ColoredVertex::new(Vec3::ZERO, Vec3::ONE.extend(0.25)));
            self.ray_lines_vertices.push(ColoredVertex::new(
                self.params_uniform.data.ray_dir,
                Vec3::ONE.extend(0.25),
            ));

            self.queue.write_buffer(
                &self.ray_lines_vertex_buffer,
                0,
                bytemuck::cast_slice(&self.ray_lines_vertices),
            );

            if self.draw_axes {
                render_pass.set_vertex_buffer(0, self.static_lines_vertex_buffer.slice(..));
                render_pass.draw(0..self.static_lines_vertices.len() as u32, 0..1);
            }

            render_pass.set_vertex_buffer(0, self.ray_lines_vertex_buffer.slice(..));
            render_pass.draw(0..self.ray_lines_vertices.len() as u32, 0..1);
        }

        // Render debug
        if self.debug_mode {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Debug Pass"),
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

            render_pass.set_bind_group(
                0,
                &self.bind_groups[&self.camera_uniform.bind_group_id],
                &[],
            );

            render_pass.set_pipeline(&self.render_pipelines[&PipelineId::Lines]);

            render_pass.set_vertex_buffer(0, self.ray_lines_vertex_buffer.slice(..));
            render_pass.draw(0..self.ray_lines_vertices.len() as u32, 0..1);
        }

        let output_view = output
            .texture
            .create_view(&TextureViewDescriptor::default());

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fullscreen Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });

            render_pass.set_bind_group(0, &self.bind_groups[&self.fullscreen_bind_group_id], &[]);

            render_pass.set_pipeline(&self.render_pipelines[&PipelineId::Fullscreen]);

            render_pass.draw(0..3, 0..1);
        }

        // Render UI
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("UI Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default()),
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

    fn render_imgui<'b>(&'b mut self, render_pass: &mut wgpu::RenderPass<'b>) -> Result<()> {
        let imgui = &mut self.imgui;

        imgui
            .platform
            .prepare_frame(imgui.context.io_mut(), &self.window)
            .expect("Failed to prepare frame");
        let ui = imgui.context.frame();

        {
            if let Some(shader) = self.hot_reload_shaders.iter().find(|shader| shader.shader_last_error.is_some()) {
                let window = ui.window(format!("Shader Error: {}", &shader.path.as_os_str().to_string_lossy()));
                window
                    .position_pivot([1.0, 0.0])
                    .position(
                        [self.config.width as f32 - 10.0, 10.0],
                        Condition::FirstUseEver,
                    )
                    .no_nav()
                    .always_auto_resize(true)
                    .focus_on_appearing(false)
                    .movable(false)
                    .save_settings(false)
                    .build(|| {
                        ui.set_window_font_scale(2.0);
                        ui.text(shader.shader_last_error.as_ref().unwrap());
                    });
            }
        }

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
                        "Camera: {:?} {:.1?} {:.1?}",
                        self.camera.position.to_array(),
                        self.camera.yaw.0.to_degrees(),
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
                    ui.checkbox("Draw Axes", &mut self.draw_axes);
                    ui.checkbox_flags(
                        "Render Wireframe",
                        &mut self.params_uniform.data.wireframe,
                        1,
                    );

                    ui.text(format!("Ray Pos: {:?}", self.params_uniform.data.ray_dir));

                    let bid = self.params_uniform.data.bid.max(0);
                    let bounces_and_length =
                        self.bounces_and_lengths_uniform.data.bounces_and_lengths[bid as usize];
                    ui.text(format!(
                        "{}->{} ({})",
                        bounces_and_length.x, bounces_and_length.y, bounces_and_length.z
                    ));

                    let max_bounces = self.lenses_uniform.data.bounce_count as i32;

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
                        self.lenses_uniform.data.interface_count as isize,
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
                        state.camera_controller.handle_mouse(delta.0, -delta.1);
                    }

                    if state.mouse_right_pressed {
                        let delta = Vec2::new(delta.1 as f32, delta.0 as f32) * 0.001;

                        let mut l = state.light_angles;

                        l.x += delta.x;
                        l.y += delta.y;
                        l.x = l.x.clamp(-PI * 0.25, PI * 0.25);

                        state.light_angles = l;
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

pub fn create_shader<'a>(
    device: &Device,
    label: Label,
    source: impl Into<Cow<'a, str>>,
) -> Result<ShaderModule> {
    device.push_error_scope(wgpu::ErrorFilter::Validation);

    let label = label.unwrap_or("Default Render Pipeline");

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some(&format!("{label} Shader")),
        source: ShaderSource::Wgsl(source.into()),
    });

    if let Some(error) = pollster::block_on(device.pop_error_scope()) {
        return Err(Error::new(error));
    }

    Ok(shader)
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
