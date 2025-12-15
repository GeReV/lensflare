use std::ops::{Add, Div, Mul, Rem, Sub};
use glam::{Vec3, Vec4};
use wgpu::{BlendComponent, BlendFactor, BlendOperation, BlendState};
use crate::colors::wavelength_to_rgb;

pub trait NumOps<Rhs = Self, Output = Self>:
    Add<Rhs, Output = Output>
    + Sub<Rhs, Output = Output>
    + Mul<Rhs, Output = Output>
    + Div<Rhs, Output = Output>
    + Rem<Rhs, Output = Output>
{
}

impl<T, Rhs, Output> NumOps<Rhs, Output> for T where
    T: Add<Rhs, Output = Output>
    + Sub<Rhs, Output = Output>
    + Mul<Rhs, Output = Output>
    + Div<Rhs, Output = Output>
    + Rem<Rhs, Output = Output>
{
}

pub fn remap<T>(v: T, from_min: T, from_max: T, to_min: T, to_max: T) -> T
where
    T: Copy + NumOps,
{
    let from_range = from_max - from_min;
    let to_range = to_max - to_min;

    (v - from_min) * (to_range / from_range) + to_min
}

pub trait Remap {
    fn remap(&self, from_min: Self, from_max: Self, to_min: Self, to_max: Self) -> Self;
}

impl<T> Remap for T where T: Copy + NumOps {
    fn remap(&self, from_min: Self, from_max: Self, to_min: Self, to_max: Self) -> Self {
        remap(*self, from_min, from_max, to_min, to_max)
    }
}

pub fn wavelengths_to_colors(wavelengths: &[f32]) -> Vec<Vec4> {
    wavelengths
        .iter()
        .map(|&wavelength| wavelength_to_rgb(wavelength).clamp(Vec3::ZERO, Vec3::ONE).extend(1.0))
        .collect()
}

pub const ADDITIVE_BLEND: BlendState = BlendState {
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