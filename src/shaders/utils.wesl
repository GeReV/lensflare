const PI: f32 = radians(180.0);

struct Params {
    debug_wireframe_alpha: f32,
    debug_interpolate: f32,
    starburst_scale: f32,
    ray_dir: vec3f,
    bid: i32,
    intensity: f32,
    lambda: f32,
    wireframe: u32,
}

fn rotation_mat2(theta: f32) -> mat2x2f {
    let c = cos(theta);
    let s = sin(theta);
    return mat2x2f(c, -s, s, c);
}