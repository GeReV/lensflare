use crate::camera::{Camera, Projection};
use crate::lenses::{Lens, LensInterface};
use crate::registry::{Id, Registry};
use crate::software::build_ray_grid_limits;
use anyhow::format_err;
use encase::internal::WriteInto;
use encase::{ArrayLength, ShaderType, UniformBuffer};
use glam::{vec3, vec4, Mat4, UVec3, Vec3, Vec4};
use std::f32::consts::PI;

const NUM_INTERFACES: usize = 32;
const NUM_BOUNCES: usize = NUM_INTERFACES * (NUM_INTERFACES - 1) / 2;

pub struct Uniform<T> {
    pub data: Box<T>,
    pub buffer_id: Id<wgpu::Buffer>,
    pub bind_group_layout_id: Id<wgpu::BindGroupLayout>,
    pub bind_group_id: Id<wgpu::BindGroup>,
}

impl<T> Uniform<T> {
    pub fn new(
        data: T,
        buffer_id: Id<wgpu::Buffer>,
        bind_group_layout_id: Id<wgpu::BindGroupLayout>,
        bind_group_id: Id<wgpu::BindGroup>,
    ) -> Self {
        Self {
            data: Box::new(data),
            buffer_id,
            bind_group_layout_id,
            bind_group_id,
        }
    }
}

impl<T> Uniform<T>
where
    T: ShaderType + WriteInto,
{
    pub fn write_buffer(
        &self,
        queue: &wgpu::Queue,
        buffers: &Registry<wgpu::Buffer>,
    ) -> Result<(), anyhow::Error> {
        queue.write_buffer(
            &buffers[&self.buffer_id],
            0,
            &UniformBuffer::<T>::content_of::<T, Vec<u8>>(&self.data)
                .map_err(|e| format_err!("Failed to transform uniform to bytes: {e}"))?,
        );

        Ok(())
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, ShaderType)]
pub struct CameraUniform {
    view_position: Vec4,
    view_proj: Mat4,
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_position: Vec4::ZERO,
            view_proj: Mat4::IDENTITY,
        }
    }

    pub fn from_camera_and_projection(camera: &Camera, projection: &Projection) -> Self {
        let mut result = Self::new();

        result.update_view_proj(camera, projection);

        result
    }

    pub fn update_view_proj(&mut self, camera: &Camera, projection: &Projection) {
        self.view_position = camera.position.extend(1.0);
        self.view_proj = (projection.calc_matrix() * camera.calc_matrix());
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, ShaderType)]
pub struct LensInterfaceUniform {
    pub center: Vec3,      // center of sphere / plane on z-axis
    pub n: Vec3,           // refractive indices (n0, n1, n2)
    pub radius: f32,       // radius of sphere/plane
    pub sa_half: f32,           // nominal radius (from optical axis)
    pub d1: f32,           // coating thickness = lambdaAR / 4 / n1
    pub flat_surface: u32, // is this interface a plane?
}

impl LensInterfaceUniform {
    pub const fn new() -> Self {
        LensInterfaceUniform {
            center: Vec3::ZERO,
            n: Vec3::ZERO,
            radius: 0.0,
            sa_half: 0.0,
            d1: 0.0,
            flat_surface: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, ShaderType)]
pub struct LensSystemUniform {
    pub interfaces: [LensInterfaceUniform; NUM_INTERFACES],
    pub interface_count: u32,
    pub bounce_count: u32,
    pub aperture_index: u32,
}

impl From<&[LensInterface]> for LensSystemUniform {
    fn from(lenses: &[LensInterface]) -> Self {
        let mut result = Self {
            interfaces: [LensInterfaceUniform::new(); NUM_INTERFACES],
            interface_count: lenses.len() as u32,
            bounce_count: {
                let bounce_lens_count = lenses.len() - 2;
                (bounce_lens_count * (bounce_lens_count - 1) / 2) as u32
            },
            aperture_index: 0,
        };

        let mut total_lens_length: f32 = 0.0;

        for (i, lens) in lenses.iter().enumerate().rev() {
            total_lens_length += lens.d;

            let n0 = if i == 0 { LensInterface::AIR_N } else { lenses[i - 1].n };
            let n2 = lens.n;
            let n1 = (n0 * n2).sqrt().max(1.38); // 1.38 = lowest achievable

            let mut n0_idx: isize = -1;
            let mut n2_idx: isize = -1;

            //if medium is not AIR, we must detect lens property
            if(n0 != 1.0 || n2 != 1.0)
            {
                let mut n0_idx_diff = f32::MAX;
                let mut n2_idx_diff = f32::MAX;


                for (idx, lens) in Lens::LENS_TABLE.iter().enumerate() {
                    if n0 != 1.0
                    {
                        let current_left_abbe_diff = (lens.nd - n0).abs();
                        if n0_idx_diff > current_left_abbe_diff
                        {
                            n0_idx = idx as isize;
                            n0_idx_diff = current_left_abbe_diff;
                        }
                    }

                    if n2 != 1.0
                    {
                        let current_right_abbe_diff = (lens.nd - n2).abs();
                        if n2_idx_diff > current_right_abbe_diff
                        {
                            n2_idx = idx as isize;
                            n2_idx_diff = current_right_abbe_diff;
                        }
                    }
                }
            }

            if lens.radius == 0.0 && i > 0 && i < result.interface_count as usize - 1 {
                result.aperture_index = i as u32;
            }

            result.interfaces[i] = LensInterfaceUniform {
                center: vec3(0.0, 0.0, total_lens_length - lens.radius),
                radius: lens.radius,
                n: vec3(n0_idx as f32, n1, n2_idx as f32),
                sa_half: lens.sa_half,
                d1: lens.coating_thickness,
                flat_surface: if lens.flat { 1 } else { 0 },
            };
        }

        result
    }
}

#[repr(C)]
#[derive(Debug, Clone, ShaderType)]
pub struct BouncesAndLengthsUniform {
    pub bounces_and_lengths: [UVec3; NUM_BOUNCES],
}

impl BouncesAndLengthsUniform {
    pub fn new(lens_count: usize, aperture_index: usize) -> Self {
        use itertools::Itertools;

        let mut vec = Vec::with_capacity(lens_count * (lens_count - 1) / 2);

        for [a, b] in (1..lens_count).array_combinations::<2>() {
            if a == aperture_index || b == aperture_index {
                continue;
            }

            // Note sure if correct: Number of bounces from entrance to b, back to a, and forward to film.
            // Assumes bounces do not count as crossing over an interface.
            let len = lens_count + 2 * (b - a) - 1;

            vec.push(UVec3::new(b as u32, a as u32, len as u32));
        }

        let mut result = Self {
            bounces_and_lengths: [UVec3::ZERO; NUM_BOUNCES],
        };

        result.bounces_and_lengths[..vec.len()].copy_from_slice(&vec);

        result
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, ShaderType)]
pub struct ParamsUniform {
    pub debug_wireframe_alpha: f32,
    pub debug_interpolate: f32,
    pub ray_dir: Vec3,
    pub bid: i32,
    pub intensity: f32,
    pub lambda: f32,
}

const N: usize = 16;

#[repr(C)]
#[derive(Debug, Clone, ShaderType)]
pub struct GridLimitsUniform {
    pub length: ArrayLength,
    #[shader(size(runtime))]
    pub limits: Vec<Vec4>,
}

impl GridLimitsUniform {
    pub fn new(
        lenses_uniform: &LensSystemUniform,
        bounces_and_lengths: &BouncesAndLengthsUniform,
    ) -> Self {
        let mut result = Self {
            length: ArrayLength,
            limits: Vec::with_capacity(NUM_BOUNCES),
        };

        for bid in 0..lenses_uniform.bounce_count as usize {
            // Start from an inverted grid (1.0-to-negative-1.0) so we can use min and max.
            let mut grid = vec4(1.0, 1.0, -1.0, -1.0);

            for y in 0..N {
                let v = y as f32 / (N - 1) as f32;

                for x in 0..N {
                    let u = x as f32 / (N - 1) as f32;

                    // if vec2(u, v).length_squared() > 1.0 {
                    //     continue;
                    // }

                    let (sin_pitch, cos_pitch) = ((0.5 - v) * PI).sin_cos();
                    let (sin_yaw, cos_yaw) = ((u - 0.5) * PI).sin_cos();


                    let ray_dir = vec3(sin_yaw * cos_pitch, sin_pitch, cos_yaw * cos_pitch).normalize();

                    let limits = build_ray_grid_limits(lenses_uniform, bounces_and_lengths, ray_dir, bid);

                    grid[0] = grid[0].max(limits[0]);
                    grid[1] = grid[1].max(limits[1]);
                    grid[2] = grid[2].min(limits[2]);
                    grid[3] = grid[3].min(limits[3]);
                }
            }

            result.limits.push(grid);
        }

        result
    }
}
