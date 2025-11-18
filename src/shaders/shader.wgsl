#import utils::{PI}
#import camera::{camera}
#import lenses::{system, vertices, indices, trace, Ray}
#import colors::{wavelength_to_rgb}

struct Params {
    debug_wireframe_alpha: f32,
    debug_interpolate: f32,
    ray_dir: vec3f,
    bid: i32,
    intensity: f32,
    lambda: f32,
    wireframe: u32,
}

@group(2) @binding(0)
var<uniform> params: Params;

//const GRID_LIMITS_SIZE: u32 = 16;
//const GRID_LIMITS_COUNT: u32 = GRID_LIMITS_SIZE * GRID_LIMITS_SIZE;
@group(2) @binding(1) var<storage, read> grid_limits: array<vec4f>;

//fn sample_grid_limits(bid: i32, ray_dir: vec3f) -> vec4f {
//    let limits = grid_limits[bid];
//
//    let size = f32(GRID_LIMITS_SIZE);
//    let xy = (ray_dir.xy + vec2f(0.5)) * (size - 1);
//    let fraction = fract(xy);
//    let index = u32(xy.y) * GRID_LIMITS_SIZE + u32(xy.x);
//
////    let a = mix(limits[index], limits[index + 1], fraction.x);
////    let b = mix(limits[index + GRID_LIMITS_SIZE], limits[index + GRID_LIMITS_SIZE + 1], fraction.x);
////
////    return mix(a, b, fraction.y);
//
//    return limits[index];
//}

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3f,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) position: vec3f,
    @location(1) color: vec4f,
    @location(2) tex: vec4f,
    @location(3) barycentric_coord: vec3f,
//    @location(1) tex_coords: vec2<f32>,
}

fn sd_hexagon(in: vec2f, r: f32) -> f32 {
    const k = vec3f(-0.866025404,0.5,0.577350269);

    var p = in;
    p = abs(p);
    p -= 2.0 * min(dot(k.xy,p),0.0) * k.xy;
    p -= vec2f(clamp(p.x, -k.z*r, k.z*r), r);

    return length(p)*sign(p.y);
}

fn edge_factor(bary: vec3f, thickness: f32) -> f32 {
  let d = fwidth(bary);
  let a3 = smoothstep(vec3f(0.0), d * thickness, bary);
  return min(min(a3.x, a3.y), a3.z);
}

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;

    let lambda = in.instance_index % 3;

    let ray_dir = normalize(params.ray_dir);

    var bid = params.bid;
    if bid < 0 {
        bid = i32(in.instance_index / 3);
    }

    let grid_limit = grid_limits[bid];

    let sa = system.interfaces[0].sa;

    let normalized_pos = in.position.xy * 2 - 1;

    var in_pos: vec2f;
    in_pos = normalized_pos;
    //in_pos = mix(grid_limit.xy, grid_limit.zw, in_pos);

    in_pos = in_pos * sa * 0.5;

    // Set lens entrance center to the center of the lens with an offset relative to the ray's position and the lens' radius,
    // i.e., radius * 0.5 away from center of the lens.
    let ray_pos = system.interfaces[0].center + vec3f(in_pos, 0);

//    let light_pos = pos * system.interfaces[0].sa;
//    let ray_dir = vec3f(0, 0, 1);
    let ray_tex = vec4f(0, 0, 0, params.intensity);

    let wavelengths = array<f32, 3>(645, 520, 408);

    let wavelength = wavelengths[lambda];

    let in_ray = Ray(ray_pos, ray_dir, ray_tex);
    let out_ray = trace(bid, in_ray, wavelength);

    var col = wavelength_to_rgb(wavelength);

//    col = vec3f();
//    switch (lambda) {
//        case 0: {
//          col.x = 1;
//        }
//        case 1: {
//          col.y = 1;
//        }
//        case 2: {
//          col.z = 1;
//        }
//        default: { }
//    }

    out.position = in.position;
    out.tex = out_ray.tex;

    var pos = mix(in_ray.pos, out_ray.pos, params.debug_interpolate);
//    pos = ;
//    if out.tex.z > 1 {
//        pos = in_pos;
//    }

    let last_lens = system.interfaces[system.interface_count - 1];

    out.clip_position = camera.view_proj * vec4f(pos.xy, last_lens.center.z, 1);
//    out.clip_position = camera.view_proj * vec4f(pos, 1.0);

//    col = vec3f(in.position.xy, 0);
//    if out.tex.z > 1 {
//        col = vec3f();
//    }

    out.color = vec4f(col, 1);

//    out.color = vec4f(f32(bounces_and_lengths[params.bid].x) / 11.0, f32(bounces_and_lengths[params.bid].y) / 11.0, 0.0, 1.0);
//    out.color = vec4f(hsv2rgb(vec3f(f32(bid) / 38.0, 0.5, 0.5)), 0.1);

//    out.color = vec4f(hsv2rgb(vec3f(out_ray.tex.a / 6, 1, 1)), 0.1);

//    out.color = vec4f(hsv2rgb(vec3f(out_ray.tex.a, 1, 1)), 0.1);

//    out.color = vec3f(1.0);

    let vert_index = in.vertex_index % 3;
    let index = indices[in.vertex_index];

    // note:
    //
    // * if your indices are U16 you could use this
    //
    //    let twoIndices = indices[vNdx / 2];  // indices is u32 but we want u16
    //    let index = (twoIndices >> ((vNdx & 1) * 16)) & 0xFFFF;
    //
    // * if you're not using indices you could use this
    //
    //    let index = vNdx;

//    let position = vec4f(vertices[index], vertices[index + 1], vertices[index + 2], 1);

    // emit a barycentric coordinate
    out.barycentric_coord = vec3f(0);
    out.barycentric_coord[vert_index] = 1.0;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    if in.tex.z > 1 {
        discard;
    }

    let focus = 22.0;
    let aperture_diameter = 8.756;
    let f = aperture_diameter / focus;

    let f_stop = f / pow(sqrt(2), 5);

    let normalized_pos = in.position.xy * 2 - 1;

//    let b = smoothstep(f_stop, f_stop - 0.01, sd_hexagon(in.tex.xy, f_stop));
    let l = length(in.tex.xy);
    let aperture = smoothstep(f_stop, f_stop - 0.01, l);

    var alpha = in.tex.a;
    if min(alpha, 1000) == 1000 {
        alpha = 0;
    }

    let ray_dir = normalize(params.ray_dir);

    var color: vec4f;
    color = vec4f(in.color.xyz, 1) * alpha * aperture;
//    color = vec4f(aperture, 0, 0, 1);

    let edge = 1.0 - edge_factor(in.barycentric_coord, 1);
    color = mix(color, vec4f(1), edge * params.debug_wireframe_alpha);
//    color = vec4f(reinhard(color);

//    color = vec4f(in.tex.xy * aperture, 0, alpha * aperture);

//    if in.tex.z > 1 {
//        color = vec4f(1);
//    }

//    if in.tex.z > 1 {
//        color.a = 0;
//    }

    return color;
}
