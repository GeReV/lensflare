mod arc;
mod buffer;
mod camera;
mod grids;
mod hot_reload;
pub mod lenses;
mod registry;
mod shaders;
mod software;
mod uniforms;
mod vertex;

use crate::arc::Arc;
use crate::camera::{Camera, CameraController, Projection};
use crate::grids::Grids;
use crate::hot_reload::{HotReloadResult, HotReloadShader};
use crate::lenses::parse_lenses;
use crate::registry::{Id, Registry};
use crate::shaders::{create_shader, init_composer};
use crate::software::calculate_grid_triangle_area_variance;
use crate::uniforms::{
    BouncesAndLengthsUniform, CameraUniform, GridLimitsUniform, ParamsUniform, Uniform,
};
use crate::vertex::{ColoredVertex, Vertex};
use anyhow::*;
use cgmath::num_traits::real::Real;
use cgmath::Angle;
use encase::{StorageBuffer, UniformBuffer};
use glam::{vec3, vec4, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use imgui::{Condition, FontSource, MouseCursor, TreeNodeFlags};
use imgui_wgpu::{Renderer, RendererConfig};
use imgui_winit_support::WinitPlatform;
use itertools::{Itertools, MinMaxResult};
use naga_oil::compose::{Composer, NagaModuleDescriptor};
use std::borrow::Cow;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::path::Path;
use std::time::{Duration, Instant};
use uniforms::LensSystemUniform;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::wgt::{SamplerDescriptor, TextureDataOrder, TextureDescriptor, TextureViewDescriptor};
use wgpu::*;
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
    Fullscreen,
}

const ADDITIVE_BLEND: BlendState = BlendState {
    color: BlendComponent {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::One,
        operation: BlendOperation::Add,
    },
    alpha: BlendComponent {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::One,
        operation: BlendOperation::Add,
    },
};

struct State {
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    is_surface_configured: bool,
    window: std::sync::Arc<Window>,
    render_pipelines: HashMap<PipelineId, RenderPipeline>,

    render_target: Texture,

    camera: Camera,
    projection: Projection,
    camera_controller: CameraController,

    composer: Composer,
    hot_reload_shaders: Vec<HotReloadShader>,

    // inputs_bind_group: BindGroup<InputsUniform>,
    // inputs_bind_group2: BindGroup<InputsUniform>,
    mouse_left_pressed: bool,
    mouse_right_pressed: bool,
    imgui: ImguiState,
    last_frame: Instant,

    buffers: Registry<Buffer>,
    bind_group_layouts: Registry<BindGroupLayout>,
    bind_groups: Registry<BindGroup>,

    default_shader_bind_group_layouts: Vec<Id<BindGroupLayout>>,

    camera_uniform: Uniform<CameraUniform>,
    lenses_uniform: Uniform<LensSystemUniform>,
    bounces_and_lengths_uniform: Uniform<BouncesAndLengthsUniform>,
    grid_limits_uniform: Uniform<GridLimitsUniform>,
    params_uniform: Uniform<ParamsUniform>,

    fullscreen_bind_group_layout_id: Id<BindGroupLayout>,
    fullscreen_bind_group_id: Id<BindGroup>,

    light_angles: Vec4,

    grids: Grids,
    bounces_grid_log2_sizes: Vec<u32>,
    grids_vertex_buffer: Buffer,
    grids_index_buffer: Buffer,

    static_lines_vertices: Vec<ColoredVertex>,
    static_lines_vertex_buffer: Buffer,
    ray_lines_vertices: Vec<ColoredVertex>,
    ray_lines_vertex_buffer: Buffer,
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

        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::VULKAN,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: None,
                required_features: Features::POLYGON_MODE_LINE,
                required_limits: Limits::default(),
                memory_hints: Default::default(),
                trace: Trace::Off,
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

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: PresentMode::Fifo, //surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let mut buffers: Registry<Buffer> = Registry::new();
        let mut bind_group_layouts: Registry<BindGroupLayout> = Registry::new();
        let mut bind_groups: Registry<BindGroup> = Registry::new();

        let render_target = device.create_texture(&TextureDescriptor {
            label: Some("Render Target Texture"),
            size: Extent3d {
                width: config.width.max(1),
                height: config.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let lenses = parse_lenses(include_str!("../lenses/wide.22mm.dat"))
            .map_err(|e| format_err!("Failed to parse lenses: {}", e))?;

        let mut lenses_uniform = LensSystemUniform::from(&lenses);

        lenses_uniform.interfaces[2].d1 = 96.4;

        // Lens at interface_count - 1 is the sensor.
        let last_lens = &lenses_uniform.interfaces[lenses_uniform.interface_count as usize];

        let view_angle = cgmath::Deg(38.0);

        let projection = Projection::new(config.width, config.height, view_angle, 0.01, 100.0);

        let (camera, camera_controller, camera_uniform) = Self::create_camera(
            &device,
            &projection,
            &mut buffers,
            &mut bind_group_layouts,
            &mut bind_groups,
            -last_lens.center.z,
        )?;

        let lenses_uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Lens System Uniform Buffer"),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            contents: &UniformBuffer::<CameraUniform>::content_of::<_, Vec<u8>>(&lenses_uniform)?,
        });

        let bounces_and_lengths_uniform = BouncesAndLengthsUniform::new(
            lenses_uniform.interface_count as usize,
            lenses_uniform.aperture_index as usize,
        );

        let bounces_and_lengths_buffer =
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Bounces and Lengths Uniform Buffer"),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                contents: &StorageBuffer::<BouncesAndLengthsUniform>::content_of::<_, Vec<u8>>(
                    &bounces_and_lengths_uniform,
                )?,
            });

        let ray_dir = -Vec3::Z;

        let lenses_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Lenses Bind Group Layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::VERTEX_FRAGMENT,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::VERTEX_FRAGMENT,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let lenses_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Lenses Bind Group"),
            layout: &lenses_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: lenses_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: bounces_and_lengths_buffer.as_entire_binding(),
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

        let grid_limits_uniform =
            GridLimitsUniform::new(&lenses_uniform.data, &bounces_and_lengths_uniform.data);

        let grids = Grids::new(4..10);

        let grid_variances = calculate_grid_triangle_area_variance(
            &lenses_uniform.data,
            &bounces_and_lengths_uniform.data,
            &grids.get_grid(1 << 4).unwrap(),
        );

        let MinMaxResult::MinMax(var_min, var_max) = grid_variances.iter().copied().minmax() else {
            panic!("Invalid variance range");
        };

        let bounces_grid_log2_sizes = {
            let grid_sizes = grids.get_grid_sizes();
            let (var_min, var_max) = (var_min.ln(), var_max.ln());

            grid_variances
                .iter()
                .map(|variance| {
                    let t = (variance.ln() - var_min) / (var_max - var_min);

                    let i = t * (grid_sizes.len() - 1) as f32;

                    grid_sizes[i as usize]
                })
                .collect_vec()
        };

        let grids_vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Grids Vertex Buffer"),
            contents: bytemuck::cast_slice(grids.vertices()),
            usage: BufferUsages::VERTEX,
        });

        let grids_index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Grids Index Buffer"),
            contents: bytemuck::cast_slice(grids.indices()),
            usage: BufferUsages::INDEX,
        });

        // Generate grids for the following 2^N range of sizes.

        // let mut img = image::Rgba32FImage::new(16, 16);
        // for (i, pixel) in img.pixels_mut().enumerate() {
        //     let limit = grid_limits_uniform.limits[28][i];
        //     *pixel = image::Rgba(limit.to_array());
        // }
        // let img8 = image::ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
        //     let p = img.get_pixel(x, y).0.map(|v| ((v + 0.5).clamp(0.0, 1.0) * 65535.0) as u16);
        //     image::Rgba(p)
        // });
        // img8.save("grid_limits.png")?;

        let grid_limits_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Grid Limits Buffer"),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            contents: &StorageBuffer::<GridLimitsUniform>::content_of::<_, Vec<u8>>(
                &grid_limits_uniform,
            )?,
        });

        let params = ParamsUniform {
            ray_dir,
            bid: 45,
            intensity: 0.5,
            lambda: 520.0, // 520 nm is some green color.
            wireframe: 0,
        };

        let params_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Tracing Params"),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            contents: &UniformBuffer::<ParamsUniform>::content_of::<_, Vec<u8>>(&params)?,
        });

        let params_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Params Bind Group Layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::VERTEX_FRAGMENT,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::VERTEX_FRAGMENT,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let params_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Params Bind Group"),
            layout: &params_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: grid_limits_buffer.as_entire_binding(),
                },
            ],
        });

        let params_bind_group_layout_id = bind_group_layouts.add(params_bind_group_layout);
        let params_bind_group_id = bind_groups.add(params_bind_group);

        let params_uniform = Uniform::new(
            params,
            buffers.add(params_buffer),
            params_bind_group_layout_id.clone(),
            params_bind_group_id.clone(),
        );

        let grid_limits_uniform = Uniform::new(
            grid_limits_uniform,
            buffers.add(grid_limits_buffer),
            params_bind_group_layout_id.clone(),
            params_bind_group_id.clone(),
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
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Lines Vertex Buffer"),
                usage: BufferUsages::VERTEX,
                contents: bytemuck::cast_slice(&static_lines_vertices),
            });

        let ray_lines_vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Lines Vertex Buffer"),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            size: (1 << 16) * size_of::<ColoredVertex>() as BufferAddress,
            mapped_at_creation: false,
        });

        let default_shader_bind_group_layouts = vec![
            camera_uniform.bind_group_layout_id.clone(),
            lenses_uniform.bind_group_layout_id.clone(),
            params_uniform.bind_group_layout_id.clone(),
        ];

        let mut composer = init_composer()?;

        let mut render_pipelines = HashMap::new();

        Self::create_static_render_pipelines(
            &device,
            &mut composer,
            &mut render_pipelines,
            &default_shader_bind_group_layouts
                .iter()
                .map(|bind_group_layout_id| &bind_group_layouts[bind_group_layout_id])
                .collect_vec(),
        );

        let mut hot_reload_shaders = Vec::new();

        hot_reload_shaders.push(HotReloadShader::new("src/shaders/shader.wgsl"));
        hot_reload_shaders.push(HotReloadShader::new("src/shaders/fullscreen.wgsl"));

        let (fullscreen_bind_group_layout_id, fullscreen_bind_group_id) = {
            let fullscreen_bind_group_layout =
                device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Render Fullscreen Bind Group Layout"),
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

            let fullscreen_bind_group_layout_id =
                bind_group_layouts.add(fullscreen_bind_group_layout);

            let sampler = device.create_sampler(&SamplerDescriptor {
                label: Some("Render Fullscreen Texture Sampler"),
                ..Default::default()
            });

            let texture_view = render_target.create_view(&TextureViewDescriptor::default());

            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("Render Fullscreen Bind Group"),
                layout: &bind_group_layouts[&fullscreen_bind_group_layout_id],
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Sampler(&sampler),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&texture_view),
                    },
                ],
            });

            let fullscreen_bind_group_id = bind_groups.add(bind_group);

            (fullscreen_bind_group_layout_id, fullscreen_bind_group_id)
        };

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

            composer,
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

            grids,
            bounces_grid_log2_sizes,
            grids_vertex_buffer,
            grids_index_buffer,

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
        projection: &Projection,
        buffers: &mut Registry<Buffer>,
        bind_group_layouts: &mut Registry<BindGroupLayout>,
        bind_groups: &mut Registry<BindGroup>,
        camera_distance: f32,
    ) -> Result<(Camera, CameraController, Uniform<CameraUniform>), anyhow::Error> {
        // For a 45 degree projection, having the same distance as the closest lens' radius should
        // have it exactly cover the "sensor".
        let camera = Camera::new(
            (0.0, 0.0, -camera_distance), // target at world origin
            cgmath::Deg(90.0),            // yaw
            cgmath::Deg(0.0),             // pitch
        );
        let camera_controller = CameraController::new(0.05, 1.75);

        let camera_uniform = CameraUniform::from_camera_and_projection(&camera, &projection);

        let camera_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Camera Buffer"),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            contents: &UniformBuffer::<CameraUniform>::content_of::<_, Vec<u8>>(&camera_uniform)?,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[BindGroupEntry {
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
        Ok((camera, camera_controller, camera_uniform))
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

    // fn build_ray_traces_debug_lines(
    //     lenses_uniform: &LensSystemUniform,
    //     bounces_and_lengths: &BouncesAndLengthsUniform,
    //     ray_grid_limits: &GridLimitsUniform,
    //     line_vertices: &mut Vec<ColoredVertex>,
    //     ray_dir: Vec3,
    //     ray_count: usize,
    //     selected_ray: Option<usize>,
    //     selected_bid: Option<usize>,
    //     ray_intensity: f32,
    //     ray_lambda: f32,
    //     ray_step_count: usize,
    // ) {
    //     let entrance = &lenses_uniform.interfaces[0];
    //
    //     let ray_range = match selected_ray {
    //         Some(ray) => ray..ray + 1,
    //         None => 0..ray_count,
    //     };
    //
    //     let selected_bid = selected_bid.unwrap_or(0);
    //
    //     let grid_limits = ray_grid_limits.limits[selected_bid];
    //
    //     for i in ray_range {
    //         let t = i as f32 / (ray_count - 1) as f32;
    //
    //         let bottom = grid_limits.br.with_x(0.0);
    //         let top = grid_limits.tl.with_x(0.0);
    //         let ray = Ray {
    //             pos: bottom + (top - bottom) * t + vec3(0.0, 0.0, 0.5),
    //             dir: ray_dir,
    //             tex: Vec4::new(0.5, t, entrance.sa, ray_intensity),
    //             hit_sensor: true,
    //         };
    //
    //         let iter = TraceIterator::new(
    //             &lenses_uniform.interfaces,
    //             &bounces_and_lengths.bounces_and_lengths,
    //             lenses_uniform.aperture_index as usize,
    //             ray,
    //             selected_bid,
    //             ray_lambda,
    //         );
    //
    //         // let color = Hsl::to_linear_srgb([t * 360., 75., 75.]);
    //         // let color = Vec3::from_array(color).extend(0.3);
    //
    //         line_vertices.extend(iter.take(ray_step_count).flat_map(|ray| {
    //             let color = if ray.intensity.is_zero() {
    //                 vec4(1.0, 0.0, 0.0, 0.05)
    //             } else {
    //                 Vec3::ONE.extend(0.2)
    //             };
    //
    //             [
    //                 ColoredVertex::new(ray.start, color),
    //                 ColoredVertex::new(ray.end, color),
    //             ]
    //         }));
    //     }
    // }

    fn create_static_render_pipelines(
        device: &Device,
        composer: &mut Composer,
        render_pipelines: &mut HashMap<PipelineId, RenderPipeline>,
        default_bind_group_layouts: &Vec<&BindGroupLayout>,
    ) {
        {
            let lines_shader = create_shader(
                device,
                composer,
                Some("Render Lines Shader"),
                Path::new("src/shaders/lines.wgsl"),
            )
            .unwrap();

            let render_pipeline_lines_layout =
                device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("Render Lines Pipeline Layout"),
                    bind_group_layouts: &[default_bind_group_layouts[0]],
                    push_constant_ranges: &[],
                });

            let targets = [Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: Some(BlendState::REPLACE),
                write_mask: ColorWrites::ALL,
            })];

            let render_pipeline_lines_desc = RenderPipelineDescriptor {
                label: Some("Render Lines Pipeline"),
                layout: Some(&render_pipeline_lines_layout),
                vertex: VertexState {
                    module: &lines_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[ColoredVertex::desc()],
                    compilation_options: PipelineCompilationOptions::default(),
                },
                fragment: Some(FragmentState {
                    module: &lines_shader,
                    entry_point: Some("fs_main"),
                    targets: &targets,
                    compilation_options: PipelineCompilationOptions::default(),
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::LineList,
                    strip_index_format: None,
                    front_face: FrontFace::Cw,
                    cull_mode: None, // Some(Face::Back),
                    // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                    polygon_mode: PolygonMode::Line,
                    // Requires Features::DEPTH_CLIP_CONTROL
                    unclipped_depth: false,
                    // Requires Features::CONSERVATIVE_RASTERIZATION
                    conservative: false,
                },
                depth_stencil: None,
                // Some(DepthStencilState {
                //     format: texture::Texture::DEPTH_FORMAT,
                //     depth_write_enabled: true,
                //     depth_compare: CompareFunction::Less,
                //     stencil: StencilState::default(),
                //     bias: DepthBiasState::default(),
                // }),
                multisample: MultisampleState {
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

    fn update_render_pipelines(&mut self) {
        for shader in self.hot_reload_shaders.iter_mut() {
            let Result::Ok(shader_result) = shader.try_hot_reload(&self.device, &mut self.composer)
            else {
                continue;
            };

            let shader_module = match shader_result {
                HotReloadResult::Updated(shader_module) => shader_module,
                HotReloadResult::Unchanged => continue,
            };

            if let Some(path) = shader.path.as_os_str().to_str() {
                match path {
                    "src/shaders/shader.wgsl" => {
                        let default_bind_group_layouts = self
                            .default_shader_bind_group_layouts
                            .iter()
                            .map(|id| &self.bind_group_layouts[id])
                            .collect_vec();

                        match Self::create_default_render_pipeline(
                            &self.device,
                            None,
                            TextureFormat::Rgba16Float,
                            &shader_module,
                            &default_bind_group_layouts,
                            None,
                        ) {
                            Result::Ok(pipeline) => {
                                shader.shader_last_error = None;

                                self.render_pipelines.insert(PipelineId::Fill, pipeline);
                            }
                            Err(err) => {
                                shader.shader_last_error = Some(err.to_string());
                            }
                        };

                        match Self::create_default_render_pipeline(
                            &self.device,
                            Some("Render Wireframe Pipeline"),
                            TextureFormat::Rgba16Float,
                            &shader_module,
                            &default_bind_group_layouts,
                            Some(PrimitiveState {
                                topology: PrimitiveTopology::LineList,
                                front_face: FrontFace::Cw,
                                cull_mode: None,
                                polygon_mode: PolygonMode::Line,
                                ..Default::default()
                            }),
                        ) {
                            Result::Ok(pipeline) => {
                                shader.shader_last_error = None;

                                self.render_pipelines
                                    .insert(PipelineId::Wireframe, pipeline);
                            }
                            Err(err) => {
                                shader.shader_last_error = Some(err.to_string());
                            }
                        };
                    }
                    "src/shaders/fullscreen.wgsl" => {
                        let render_pipeline_layout =
                            self.device
                                .create_pipeline_layout(&PipelineLayoutDescriptor {
                                    label: Some("Render Fullscreen Pipeline Layout"),
                                    bind_group_layouts: &[&self.bind_group_layouts
                                        [&self.fullscreen_bind_group_layout_id]],
                                    push_constant_ranges: &[],
                                });

                        let render_pipeline_fullscreen =
                            self.device
                                .create_render_pipeline(&RenderPipelineDescriptor {
                                    label: Some("Render Fullscreen Pipeline"),
                                    layout: Some(&render_pipeline_layout),
                                    vertex: VertexState {
                                        module: &shader_module,
                                        entry_point: Some("vs_main"),
                                        buffers: &[],
                                        compilation_options:
                                            PipelineCompilationOptions::default(),
                                    },
                                    fragment: Some(FragmentState {
                                        module: &shader_module,
                                        entry_point: Some("fs_main"),
                                        targets: &[Some(ColorTargetState {
                                            format: self.config.format,
                                            blend: Some(BlendState::REPLACE),
                                            write_mask: ColorWrites::ALL,
                                        })],
                                        compilation_options:
                                            PipelineCompilationOptions::default(),
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

                        self.render_pipelines
                            .insert(PipelineId::Fullscreen, render_pipeline_fullscreen);
                    }
                    _ => {}
                }
            }
        }
    }

    fn create_default_render_pipeline<'shader>(
        device: &Device,
        label: Label<'shader>,
        texture_format: TextureFormat,
        shader: &ShaderModule,
        bind_group_layouts: &[&BindGroupLayout],
        primitive: Option<PrimitiveState>,
    ) -> Result<RenderPipeline> {
        let label = label.unwrap_or("Default Render Pipeline");

        let render_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some(&format!("{label} Layout")),
                bind_group_layouts: &bind_group_layouts,
                push_constant_ranges: &[],
            });

        let render_pipeline_default = RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: texture_format,
                    blend: Some(ADDITIVE_BLEND),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: primitive.unwrap_or(PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Cw,
                cull_mode: None, // Some(Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            }),
            depth_stencil: None,
            multisample: MultisampleState {
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
        device: &Device,
        queue: &Queue,
        surface_format: TextureFormat,
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

        self.lenses_uniform
            .write_buffer(&self.queue, &self.buffers)
            .unwrap();

        self.params_uniform.data.ray_dir = -ray_pos;

        self.params_uniform
            .write_buffer(&self.queue, &self.buffers)
            .unwrap();

        self.update_render_pipelines();

        self.ray_lines_vertices.clear();

        if self.debug_mode {
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

            // let lens_center = self.lenses_uniform.data.interfaces[0].center;

            // let ray_dir = (lens_center - self.params_uniform.data.ray_dir).normalize();

            // Self::build_ray_traces_debug_lines(
            //     &self.lenses_uniform.data,
            //     &self.bounces_and_lengths_uniform.data,
            //     &self.grid_limits_uniform.data,
            //     &mut self.ray_lines_vertices,
            //     ray_dir,
            //     DEBUG_RAY_COUNT as usize,
            //     selected_ray,
            //     selected_bid,
            //     1.0,
            //     520.0,
            //     self.ray_step_count,
            // );
        }

        {
            self.ray_lines_vertices
                .push(ColoredVertex::new(Vec3::ZERO, Vec3::ONE.extend(0.25)));
            self.ray_lines_vertices.push(ColoredVertex::new(
                self.params_uniform.data.ray_dir,
                Vec3::ONE.extend(0.25),
            ));

            let lens = &self.lenses_uniform.data.interfaces
                [self.lenses_uniform.data.interface_count as usize - 1];
            let arc_verts = Arc::circle(lens.center.xy(), lens.sa * 0.5)
                .iter(32)
                .collect::<Vec<_>>();

            let arc_verts = arc_verts[..]
                .windows(2)
                .flat_map(|a| {
                    [
                        ColoredVertex::new(
                            Vec3::new(a[0].x, a[0].y, lens.center.z),
                            Vec3::Z.extend(0.2),
                        ),
                        ColoredVertex::new(
                            Vec3::new(a[1].x, a[1].y, lens.center.z),
                            Vec3::Z.extend(0.2),
                        ),
                    ]
                })
                .collect::<Vec<_>>();

            self.ray_lines_vertices.extend(arc_verts);
        }
        if !self.ray_lines_vertices.is_empty() {
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
            .create_view(&TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let output = self.surface.get_current_texture()?;

        // Render flare
        if !self.debug_mode {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Lens Flare Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
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

            let pipeline = if self.params_uniform.data.wireframe == 1 {
                PipelineId::Wireframe
            } else {
                PipelineId::Fill
            };

            render_pass.set_vertex_buffer(0, self.grids_vertex_buffer.slice(..));
            render_pass
                .set_index_buffer(self.grids_index_buffer.slice(..), IndexFormat::Uint32);

            if let Some(render_pipeline) = self.render_pipelines.get(&pipeline) {
                render_pass.set_pipeline(render_pipeline);

                let selected_bid = self.params_uniform.data.bid;
                let bounce_range = if selected_bid < 0 {
                    0..self.lenses_uniform.data.bounce_count
                } else {
                    selected_bid as u32..(selected_bid + 1) as u32
                };

                for bid in bounce_range {
                    let grid_size = self.bounces_grid_log2_sizes[bid as usize] as usize;

                    let (_, index_range) =
                        self.grids.get_grid_size_index_ranges(grid_size).unwrap();

                    render_pass.draw_indexed(index_range, 0, (bid * 3)..(bid + 1) * 3);
                }

                self.params_uniform.data.bid = selected_bid;
            }
        }

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Lines Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
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

            if self.draw_axes {
                render_pass.set_vertex_buffer(0, self.static_lines_vertex_buffer.slice(..));
                render_pass.draw(0..self.static_lines_vertices.len() as u32, 0..1);
            }

            render_pass.set_vertex_buffer(0, self.ray_lines_vertex_buffer.slice(..));
            render_pass.draw(0..self.ray_lines_vertices.len() as u32, 0..1);
        }

        // Render debug
        if self.debug_mode {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Debug Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
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
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Fullscreen Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &output_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
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
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("UI Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &output
                        .texture
                        .create_view(&TextureViewDescriptor::default()),
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
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

    fn render_imgui<'b>(&'b mut self, render_pass: &mut RenderPass<'b>) -> Result<()> {
        let imgui = &mut self.imgui;

        imgui
            .platform
            .prepare_frame(imgui.context.io_mut(), &self.window)
            .expect("Failed to prepare frame");
        let ui = imgui.context.frame();

        {
            if let Some(shader) = self
                .hot_reload_shaders
                .iter()
                .find(|shader| shader.shader_last_error.is_some())
            {
                let window = ui.window(format!(
                    "Shader Error: {}",
                    &shader.path.as_os_str().to_string_lossy()
                ));
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

                    // ui.separator();
                    // let vec = self.grid_variances.1.iter().copied().map(|x| x.log2()).sorted_by(|a, b| a.total_cmp(b)).collect_vec();
                    // ui.plot_lines("Grid Triangles Area Variance", &vec)
                    //     .build();

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

                    let max_bounces = (self.lenses_uniform.data.bounce_count - 1) as i32;

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

                    // List of slider for the anti-reflection coating on each interface.
                    if ui.collapsing_header("Anti-reflection Coatings", TreeNodeFlags::DEFAULT_OPEN)
                    {
                        for i in 1..self.lenses_uniform.data.interface_count {
                            let interface = &mut self.lenses_uniform.data.interfaces[i as usize];

                            ui.slider(
                                format!("Interface {i}"),
                                0.0,
                                830.0 * 0.25,
                                &mut interface.d1,
                            );
                        }
                    }
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
                    match err.downcast_ref::<SurfaceError>() {
                        Some(SurfaceError::Lost | SurfaceError::Outdated) => {
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
                        let delta = Vec2::new(-delta.1 as f32, delta.0 as f32) * 0.001;

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
