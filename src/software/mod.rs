use crate::uniforms::LensInterfaceUniform as LensInterface;
use glam::{UVec3, Vec3, Vec3Swizzles, Vec4};
use std::f32::consts::PI;

pub mod trace_iterator;

#[derive(Clone)]
pub struct Ray {
    pub(crate) pos: Vec3,
    pub(crate) dir: Vec3,
    pub(crate) tex: Vec4,
}

#[derive(Default)]
struct Intersection {
    pos: Vec3,
    norm: Vec3,
    theta: f32,     // incident angle
    hit: bool,      // intersection found?
    inverted: bool, // inverted intersection?
}

fn reflect(e1: Vec3, e2: Vec3) -> Vec3 {
    e1 - 2.0 * e2.dot(e1) * e2
}

fn refract(e1: Vec3, e2: Vec3, n: f32) -> Vec3 {
    let k = 1.0 - n * n * (1.0 - e2.dot(e1) * e2.dot(e1));
    if k < 0.0 {
        return Vec3::ZERO;
    }

    n * e1 - (n * e2.dot(e1) + k.sqrt()) * e2
}

fn fresnel_anti_reflect(
    theta0: f32, // angle of incidence
    lambda: f32, // wavelength of ray
    d1: f32,     // thickness of anti-reflection coating
    n0: f32,     // RI (refr. index) of 1st medium
    n1: f32,     // RI of coating layer
    n2: f32,     // RI of the 2nd medium
) -> f32 {
    // refraction angle sin coating and the 2nd medium
    let theta1 = (theta0.sin() * n0 / n1).asin();
    let theta2 = (theta0.sin() * n0 / n2).asin();

    // amplitude for outer refl./transmission on topmost interface
    let rs01 = -(theta0 - theta1).sin() / (theta0 + theta1).sin();
    let rp01 = (theta0 - theta1).tan() / (theta0 + theta1).tan();
    let ts01 = 2.0 * (theta1.sin()) * (theta0).cos() / (theta0 + theta1).sin();
    let tp01 = ts01 * (theta0 - theta1).cos();

    // amplitude for inner reflection
    let rs12 = -(theta1 - theta2).sin() / (theta1 + theta2).sin();
    let rp12 = (theta1 - theta2).tan() / (theta1 + theta2).tan();

    // after passing through first surface twice:
    // 2 transmissions and 1 reflection
    let ris = ts01 * ts01 * rs12;
    let rip = tp01 * tp01 * rp12;

    // phase difference between outer and inner reflections
    let dy = d1 * n1;
    let dx = (theta1.tan()) * dy;
    let delay = (dx * dx + dy * dy).sqrt();
    let rel_phase = 4.0 * PI / lambda * (delay - dx * (theta0.sin()));

    // Add up sines of different phase and amplitude
    let out_s2 = rs01 * rs01 + ris * ris + 2.0 * rs01 * ris * rel_phase.cos();
    let out_p2 = rp01 * rp01 + rip * rip + 2.0 * rp01 * rip * rel_phase.cos();

    (out_s2 + out_p2) / 2.0 // reflectivity
}

fn test_flat(r: &Ray, f: &LensInterface) -> Intersection {
    Intersection {
        pos: r.pos + r.dir * ((f.center.z - r.pos.z) / r.dir.z),
        theta: 0., // meaningless
        hit: true,
        inverted: false,
        norm: if r.dir.z > 0.0 {
            Vec3::new(0., 0., -1.)
        } else {
            Vec3::new(0., 0., 1.)
        },
    }
}

fn test_sphere(r: &Ray, f: &LensInterface) -> Intersection {
    let mut i = Intersection::default();

    let d = r.pos - f.center;
    let b = d.dot(r.dir);
    let c = d.dot(d) - f.radius * f.radius;
    let b2_c = b * b - c;

    if b2_c < 0.0 {
        // no intersection
        i.hit = false;
        return i;
    }

    let sgn = if f.radius * r.dir.z > 0.0 { 1.0 } else { -1.0 };
    let t = b2_c.sqrt() * sgn - b;

    i.pos = r.dir * t + r.pos;
    i.norm = (i.pos - f.center).normalize();

    if i.norm.dot(r.dir) > 0.0 {
        i.norm = -i.norm;
    }

    i.theta = -r.dir.dot(i.norm).acos();
    i.hit = true;
    i.inverted = t < 0.0; // mark an inverted ray

    i
}

pub fn trace(
    bid: usize,  // index of current bounce/ghost
    r_in: &Ray,  // input ray from the entrance plane
    lambda: f32, // wavelength of ray
    interfaces: &[LensInterface],
    bounces_and_lengths: &[UVec3],
    aperture_index: usize,
) -> Ray {
    let mut r: Ray = r_in.clone();

    let surfaces = bounces_and_lengths[bid].xy().to_array(); // read 2 surfaces to reflect
    let length = bounces_and_lengths[bid].z; // length of intersections

    println!("{surfaces:?}");

    // initialization
    let mut phase: usize = 0; // ray-tracing phase
    let mut delta: isize = 1; // delta for for-loop
    let mut t: usize = 1; // index of target in trace to test

    let mut k: usize = 0;
    while k < length as usize {
        let f = interfaces[t];
        let b_reflect = if phase < 2 {
            t == surfaces[phase] as usize
        } else {
            false
        };
        if b_reflect {
            delta = -delta;
            phase += 1;
        }

        // i n t e r s e c t i o n t e s t
        let i = if f.flat_surface == 1 {
            test_flat(&r, &f)
        } else {
            test_sphere(&r, &f)
        };

        if !i.hit {
            // exit upon miss
            break;
        }

        // record texture coord. or max. rel. radius
        if f.flat_surface == 0 {
            r.tex.z = r.tex.z.max(i.pos.xy().length() / f.sa);
        } else if t == aperture_index {
            // iris aperture plane
            r.tex.x = i.pos.x / interfaces[aperture_index].radius;
            r.tex.y = i.pos.y / interfaces[aperture_index].radius;
        }

        // update ray direction and position
        r.dir = (i.pos - r.pos).normalize();

        if i.inverted {
            r.dir *= -1.0; // corrected inverted ray
        }

        r.pos = i.pos;

        // skip reflection/refraction for flat surfaces
        if f.flat_surface == 1 {
            continue;
        }

        // do reflection/refraction for spher. surfaces
        let n0: f32 = if r.dir.z < 0.0 { f.n.x } else { f.n.z };
        let n1 = f.n.y;
        let n2: f32 = if r.dir.z < 0.0 { f.n.x } else { f.n.z };

        if !b_reflect
        {
            // refraction
            r.dir = refract(r.dir, i.norm, n0 / n2);
            if r.dir == Vec3::ZERO {
                // total reflection
                break;
            }
        } else
        {
            // reflection with anti-reflection coating
            r.dir = reflect(r.dir, i.norm);
            let anti_ref = fresnel_anti_reflect(i.theta, lambda, f.d1, n0, n1, n2);
            r.tex.w *= anti_ref; // update ray intensity
        }

        t = t.checked_add_signed(delta).unwrap();
        k += 1;
    }

    if k < length as usize {
        r.tex.w = 0.0; // early-exit rays = invalid
        println!("{k}/{length}");
    } else {
        println!("trace");
    }

    r
}

