const NUM_INTERFACES: u32 = 32;

struct Lens
{
    a0: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    a4: f32,
    a5: f32,
    a6: f32,
    a7: f32,
    a8: f32,
    nc: f32,
    nd: f32,
    nf: f32,
}

struct LensInterface {
    center: vec3f, // center of sphere / plane on z-axis
    n: vec2u, // indices of lens in array of lens types
    radius: f32, // radius of sphere/plane
    sa_half: f32, // nominal radius (from optical axis)
    d1: f32, // coating thickness = lambdaAR / 4 / n1
    flat_surface: u32, // is this interface a plane?
}

struct Ray {
    pos: vec3f,
    dir: vec3f,
    tex: vec4f,
}

struct Intersection {
    pos: vec3f,
    norm: vec3f,
    theta: f32, // incident angle
    hit: bool, // intersection found?
    inverted: bool, // inverted intersection?
}

struct LensSystem {
    interfaces: array<LensInterface, NUM_INTERFACES>,
    interface_count: u32,
    bounce_count: u32,
    aperture_index: i32,
}
