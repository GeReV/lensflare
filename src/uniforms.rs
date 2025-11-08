use crate::camera::{Camera, Projection};
use crate::lenses::LensInterface;
use crate::registry::{Id, Registry};
use anyhow::format_err;
use cgmath::Zero;
use encase::internal::WriteInto;
use encase::{ShaderType, UniformBuffer};
use glam::{Mat4, UVec3, Vec3, Vec4};

const NUM_INTERFACES: usize = 64;
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
    pub sa: f32,           // nominal radius (from optical axis)
    pub d1: f32,           // coating thickness = lambdaAR / 4 / n1
    pub flat_surface: u32, // is this interface a plane?
}

impl LensInterfaceUniform {
    pub const fn new() -> Self {
        LensInterfaceUniform {
            center: Vec3::ZERO,
            n: Vec3::ZERO,
            radius: 0.0,
            sa: 0.0,
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

impl From<&Vec<LensInterface>> for LensSystemUniform {
    fn from(lenses: &Vec<LensInterface>) -> Self {
        let mut result = Self {
            interfaces: [LensInterfaceUniform::new(); 64],
            interface_count: 1 + lenses.len() as u32,
            bounce_count: {
                let bounce_lens_count = lenses.len() - 1;
                (bounce_lens_count * (bounce_lens_count - 1) / 2) as u32
            },
            aperture_index: 0,
        };

        let mut prev_refr_index = 1.0;
        let mut offset = 0.0;
        let mut prev_thickness = 0.0;

        const SCALE_FACTOR: f32 = 1e-2;

        result.interfaces[0] = LensInterfaceUniform {
            center: Vec3::ZERO,
            radius: lenses[0].radius * SCALE_FACTOR,
            n: Vec3::ONE,
            sa: lenses[0].aperture * SCALE_FACTOR,
            d1: 0.0,
            flat_surface: 1,
        };

        for (i, lens) in lenses.iter().enumerate() {
            let n0 = prev_refr_index;
            let n2 = lens.n;
            let n1 = (n0 * n2).sqrt().max(1.38); // 1.38 = lowest achievable

            let radius = lens.radius * SCALE_FACTOR;

            offset += result.interfaces[i].radius;
            offset -= radius;
            offset -= prev_thickness;

            result.interfaces[i + 1] = LensInterfaceUniform {
                center: Vec3::new(0.0, 0.0, offset),
                radius,
                n: Vec3::new(n0, n1, n2),
                sa: lens.aperture * SCALE_FACTOR,
                d1: 0.0, // TODO: lambda0 / 4 / n1 ; // phase delay
                flat_surface: 0,
            };

            if lens.radius.is_zero() {
                // This is the aperture.
                result.interfaces[i + 1].flat_surface = 1;
                result.interfaces[i + 1].n = Vec3::ONE;

                result.aperture_index = (i + 1) as u32;
            }

            prev_refr_index = result.interfaces[i + 1].n.z;
            prev_thickness = lens.axis_position * SCALE_FACTOR;
        }

        result
    }
}

#[repr(C)]
#[derive(Debug, Clone, ShaderType)]
pub struct BouncesAndLengthsUniform {
    pub data: [UVec3; NUM_BOUNCES],
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
            data: [UVec3::ZERO; NUM_BOUNCES],
        };

        result.data[..vec.len()].copy_from_slice(&vec);

        result
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, ShaderType)]
pub struct ParamsUniform {
    pub light_pos: Vec3,
    pub bid: i32,
    pub intensity: f32,
    pub lambda: f32,
    pub wireframe: u32,
}
