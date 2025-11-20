use glam::{UVec3, Vec3, Vec3Swizzles};
use crate::software;
use crate::software::Ray;
use crate::uniforms::LensInterfaceUniform as LensInterface;

impl<'a> TraceIterator<'a> {
    pub fn new(lens_interfaces: &'a [LensInterface], bounces_and_lengths: &'a [UVec3], aperture_index: usize, ray: Ray, bid: usize, lambda: f32) -> Self {
        Self {
            lens_interfaces,
            bounces_and_lengths,
            aperture_index,
            bid,
            ray,
            lambda,
            phase: 0,
            delta: 1,
            t: 1,
            k: 0,
        }
    }
}

pub struct TraceIterator<'a> {
    lens_interfaces: &'a [LensInterface],
    bounces_and_lengths: &'a [UVec3],
    aperture_index: usize,
    bid: usize,
    ray: Ray,
    lambda: f32,
    phase: usize,
    delta: isize,
    t: usize,
    k: usize,
}

pub struct RaySegment {
    pub start: Vec3,
    pub end: Vec3,
    pub intensity: f32,
    pub progress: f32,
}

impl<'a> Iterator for TraceIterator<'a> {
    type Item = RaySegment;

    fn next(&mut self) -> Option<Self::Item> {
        let surfaces = &self.bounces_and_lengths[self.bid].xy(); // read 2 surfaces to reflect
        let length = self.bounces_and_lengths[self.bid].z as usize; // length of intersections

        let mut result = RaySegment {
            start: self.ray.pos,
            end: Vec3::ZERO,
            intensity: 1.0,
            progress: self.k as f32 / length as f32,
        };

        // initialization
        while self.k < length {
            let f = self.lens_interfaces[self.t];
            let b_reflect = if self.phase < 2 {
                self.t == surfaces[self.phase] as usize
            } else {
                false
            };
            if b_reflect {
                self.delta = -self.delta;
                self.phase += 1;
            }

            // i n t e r s e c t i o n t e s t
            let i = if f.flat_surface == 1 {
                software::test_flat(&self.ray, &f)
            } else {
                software::test_sphere(&self.ray, &f)
            };

            if !i.hit {
                // exit upon miss
                self.k += 1;
                break;
            }

            // record texture coord. or max. rel. radius
            if f.flat_surface == 0 {
                self.ray.tex.z = self.ray.tex.z.max(i.pos.xy().length() / f.sa_half);
            } else if self.t == self.aperture_index {
                // iris aperture plane
                self.ray.tex.x = i.pos.x / self.lens_interfaces[self.aperture_index].sa_half;
                self.ray.tex.y = i.pos.y / self.lens_interfaces[self.aperture_index].sa_half;
            }

            // update ray direction and position
            self.ray.dir = (i.pos - self.ray.pos).normalize();

            if i.inverted {
                self.ray.dir *= -1.0; // corrected inverted ray
            }

            self.ray.pos = i.pos;
            result.end = self.ray.pos;

            // skip reflection/refraction for flat surfaces
            if f.flat_surface == 1 {
                self.t = self.t.checked_add_signed(self.delta).unwrap();
                self.k += 1;

                return Some(result);
            }

            // do reflection/refraction for spher. surfaces
            let n1 = f.n.y;

            let (n0, n2) = if self.ray.dir.z < 0.0 {
                (f.n.x, f.n.z)
            } else {
                (f.n.z, f.n.x)
            };

            if !b_reflect
            {
                // refraction
                self.ray.dir = software::refract(self.ray.dir, i.norm, n0 / n2);
                if self.ray.dir == Vec3::ZERO {
                    // total reflection
                    self.k += 1;
                    break;
                }
            } else {
                // reflection with anti-reflection coating
                self.ray.dir = software::reflect(self.ray.dir, i.norm);
                let anti_ref = software::fresnel_anti_reflect(i.theta, self.lambda, f.d1, n0, n1, n2);
                self.ray.tex.w *= anti_ref; // update ray intensity

                result.intensity = self.ray.tex.w;
            }

            self.t = self.t.checked_add_signed(self.delta).unwrap();
            self.k += 1;

            return Some(result);
        }

        if length > 0 && self.k == length {
            self.k = length + 1; // Mark end of iterations

            return Some(result);
        } else if self.k < length {
            self.ray.tex.w = 0.0; // early-exit rays = invalid

            result.intensity = self.ray.tex.w;

            self.k = length + 1;

            return Some(result);
        }

        None
    }
}