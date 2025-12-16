import super::utils::{PI, Params};
import super::camera::camera;
import super::lenses_types::Ray;
import super::lenses::{trace, system};
import super::colors::wavelength_to_rgb;

@group(2) @binding(0)
var<uniform> params: Params;

@group(3) @binding(0) var ghost_sampler : sampler;
@group(3) @binding(1) var ghost_texture : texture_2d<f32>;

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
    @location(1) color: vec3f,
    @location(2) tex: vec4f,
    @location(3) scale: f32,
//    @location(3) barycentric_coord: vec3f,
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

// A quad ABCD's area is calculated as the area of the two triangles making it.
// Each triangle's area is calculated as half of the magnitude of cross-product of its edges.
// The magnitude of a cross-product between two vectors is the area of the parallelogram created by AB and AC.
// i.e. for triangle ABC, |ABxAC| / 2.
fn quad_area(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> f32 {
    let ab = length(a - b);
    let ac = length(a - c);
    let bd = length(b - d);
    let cd = length(c - d);

    return (length(cross(ab, ac)) + length(cross(bd, cd))) * 0.5;
}

fn calculate_grid_start_index(vertex_index: u32) -> u32 {
    var i: u32 = 0;
    var prev_start_index: u32 = 0;

    for (; i < 32; i += 1) {
        let grid_size = u32(1 << i) + 1;

        let start_index = grid_size * grid_size;

        if start_index >= vertex_index {
            break;
        }

        prev_start_index = start_index;
    }

    return prev_start_index;
}

fn calculate_grid_scale(vertex_index: u32) -> f32 {
    let grid_start_index = calculate_grid_start_index(vertex_index);
    let grid_size = sqrt(grid_start_index);

    let base_cell_width = 2 / f32(grid_size);
    let base_cell_area = base_cell_width * base_cell_width;

    // The 9 vertices around and including vertex_index.
    let a = vertices[vertex_index - grid_size - 1];
    let b = vertices[vertex_index - grid_size + 0];
    let c = vertices[vertex_index - grid_size + 1];
    let d = vertices[vertex_index - 1];
    let e = vertices[vertex_index + 0];
    let f = vertices[vertex_index + 1];
    let g = vertices[vertex_index + grid_size - 1];
    let h = vertices[vertex_index + grid_size + 0];
    let i = vertices[vertex_index + grid_size + 1];

}

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;

    let lambda = in.instance_index % 3;

    let last_lens = system.interfaces[system.interface_count - 1];

    let ray_dir = normalize(params.ray_dir);

    var bid = params.bid;
    if bid < 0 {
        bid = i32(in.instance_index / 3);
    }

    let normalized_pos = in.position.xy * 2 - 1;
    let sa = last_lens.sa_half;

    var in_pos: vec2f;
    in_pos = normalized_pos;

//    in_pos *= sa;

    // Set lens entrance center to the center of the lens with an offset relative to the ray's position and the lens' radius,
    // i.e., radius * 0.5 away from center of the lens.
    let ray_pos = system.interfaces[0].center + vec3f(in_pos, 0);

//    let light_pos = pos * system.interfaces[0].sa;
//    let ray_dir = vec3f(0, 0, 1);
    let ray_tex = vec4f(0, 0, 0, 1);

    let wavelengths = array<f32, 3>(645, 520, 408);

    let wavelength = wavelengths[lambda];

    let in_ray = Ray(ray_pos, ray_dir, ray_tex);
    let out_ray = trace(bid, in_ray, wavelength);

    var col = wavelength_to_rgb(wavelength);

    out.position = in.position;
    out.tex = out_ray.tex;

    var out_pos = out_ray.pos;
    if out_pos.z == 1 {
        out_pos.z = last_lens.center.z;
    }

    var pos = mix(in_ray.pos, out_pos, params.debug_interpolate);

    out.clip_position = camera.view_proj * vec4f(pos.xyz, 1);
//    out.clip_position = camera.view_proj * vec4f(pos, 1.0);

//    col = vec3f(in.position.xy, 0);
//    if out.tex.z > 1 {
//        col = vec3f();
//    }

    out.color = col;

//    out.scale = calculate_grid_scale(in.vertex_index);

//    if out_pos.x < -1 || out_pos.x > 1 || out_pos.y < -1 || out_pos.y > 1 {
//        out.tex.a = 0;
//    }

//    out.color = vec4f(f32(bounces_and_lengths[params.bid].x) / 11.0, f32(bounces_and_lengths[params.bid].y) / 11.0, 0.0, 1.0);
//    out.color = vec4f(hsv2rgb(vec3f(f32(bid) / 38.0, 0.5, 0.5)), 0.1);

//    out.color = vec4f(hsv2rgb(vec3f(out_ray.tex.a / 6, 1, 1)), 0.1);

//    out.color = vec4f(hsv2rgb(vec3f(out_ray.tex.a, 1, 1)), 0.1);

//    out.color = vec3f(1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    if in.tex.z > 1 || in.tex.a == 0 {
        discard;
    }

    let focus = 25.0;
    let aperture_diameter = 14.0;
    let f = aperture_diameter / focus;

    let f_stop = f / pow(sqrt(2), 5);

    let normalized_pos = in.position.xy * 2 - 1;

    let alpha = in.tex.a;

    let uv = (in.tex.xy / f_stop + 0.5);
    let tex = textureSample(ghost_texture, ghost_sampler, uv);

    let vignette = smoothstep(0.1, 0.5, 1 - length(in.position.xy * 2 - 1));

    let color = vec4f(in.color.xyz, 1);

    return color * tex * vignette * alpha * params.intensity;
}
