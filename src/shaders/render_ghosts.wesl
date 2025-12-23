import super::utils::{PI, Params};
import super::camera::camera;
import super::lenses_types::Ray;
import super::lenses::{trace, system, test_sphere};
import super::colors::wavelength_to_rgb;
import constants::{workgroup_size_x, workgroup_size_y, workgroup_size_z, GRID_SIZE};

@group(2) @binding(0)
var<uniform> params: Params;

@group(3) @binding(0) var ghost_sampler : sampler;
@group(3) @binding(1) var ghost_texture : texture_2d<f32>;

@if (compute)
@group(4) @binding(0) var<storage, read_write> rays: array<Ray>;
@else
@group(4) @binding(0) var<storage, read> rays: array<Ray>;

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) position: vec3f,
    @location(1) color: vec3f,
    @location(2) tex: vec4f,
    @location(3) barycentric_coord: vec3f,
    @location(4) tex_coords: vec2f,
    @location(5) area_ratio: f32,
}

const SPREAD: f32 = 0.75;

const WAVELENGTHS: array<f32, 3> = array(645, 520, 408);

fn edge_factor(x: vec3f) -> f32 {
  let d = fwidth(x);
  let f = smoothstep(vec3f(), d, x);
  return min(min(f.x, f.y), f.z);
}

// A quad ABCD's area is calculated as the area of the two triangles making it.
// Each triangle's area is calculated as half of the magnitude of cross-product of its edges.
// The magnitude of a cross-product between two vectors is the area of the parallelogram created by AB and AC.
// i.e. for triangle ABC, |ABxAC| / 2.
fn quad_area(a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> f32 {
    let ab = a - b;
    let ac = a - c;
    let bd = b - d;
    let cd = c - d;

    return (length(cross(ab, ac)) + length(cross(bd, cd))) * 0.5;
}

fn calculate_grid_cell_area_ratio(vertex_index: u32) -> f32 {
    // The 9 vertices around and including vertex_index.
    let a = rays[vertex_index - GRID_SIZE - 1];
    let b = rays[vertex_index - GRID_SIZE + 0];
    let c = rays[vertex_index - GRID_SIZE + 1];
    let d = rays[vertex_index - 1];
    let e = rays[vertex_index + 0];
    let f = rays[vertex_index + 1];
    let g = rays[vertex_index + GRID_SIZE - 1];
    let h = rays[vertex_index + GRID_SIZE + 0];
    let i = rays[vertex_index + GRID_SIZE + 1];

    var base_cell_width = SPREAD / f32(GRID_SIZE - 1);
    var base_cell_height = base_cell_width;

    let A = quad_area(a.pos, b.pos, d.pos, e.pos);
    let B = quad_area(b.pos, c.pos, e.pos, f.pos);
    let C = quad_area(d.pos, e.pos, g.pos, h.pos);
    let D = quad_area(e.pos, f.pos, h.pos, i.pos);

    var area = D;

    let cell_pos = vec2u(vertex_index % GRID_SIZE, vertex_index / GRID_SIZE);

    if cell_pos.x > 0 && cell_pos.x < GRID_SIZE - 1 {
        base_cell_width *= 2;

        area += C;
    }
    if cell_pos.y > 0 && cell_pos.y < GRID_SIZE - 1 {
        base_cell_height *= 2;

        area += B;
    }

    if all(cell_pos > vec2u()) && all(cell_pos < vec2u(GRID_SIZE - 1)) {
        area += A;
    }

    let base_cell_area = base_cell_width * base_cell_height;

    let energy = 4.0;
    return max(0, (base_cell_area / (area + 0.00001)) * energy);
}

// TODO: Check lens system setup compared to other implementation

@if (compute)
@compute
@workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)
fn trace_rays(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u
) {
    var bid = params.bid;
    if bid < 0 {
        bid = i32(gid.z);
    }

//    if u32(bid) >= system.bounce_count {
//        return;
//    }

    let first_lens = system.interfaces[0];

    let pos = vec2u(gid.x % GRID_SIZE, gid.x / GRID_SIZE);
    let uv = vec2f(pos) / vec2f(GRID_SIZE - 1);
    let normalized_uv = SPREAD * (uv * 2 - 1);

    let test_ray_pos = vec3f(normalized_uv, 1000);
    let test_ray = Ray(test_ray_pos, vec3f(0, 0, -1), vec4f(0));
    let first_lens_intersection = test_sphere(test_ray, first_lens);

    let ray_pos = first_lens_intersection.pos - params.ray_dir;
    let ray_dir = normalize(params.ray_dir);
    let ray_tex = vec4f(0, 0, 0, 1);

    let wavelength = WAVELENGTHS[gid.y % 3];

    let in_ray = Ray(ray_pos, ray_dir, ray_tex);
    let out_ray = trace(bid, in_ray, wavelength);

    let grid_total_count = u32(GRID_SIZE * GRID_SIZE);
    let index = gid.x + gid.y * grid_total_count + gid.z * grid_total_count * 3;

    rays[index] = out_ray;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let GS = u32(GRID_SIZE);
    let grid_total_count = GS * GS;

    let vertices: array<u32, 6> = array(
        0,
        1,
        GS,
        1,
        GS + 1,
        GS,
    );

    let cell_idx = in.vertex_index / 6;
    let cell_coord = vec2u(cell_idx % (GS - 1), cell_idx / (GS - 1));
    let cell_corner_vertex_idx = cell_coord.x + cell_coord.y * GS;
    let grid_local_vertex_idx = cell_corner_vertex_idx + vertices[in.vertex_index % 6];

    let index = in.instance_index * grid_total_count + grid_local_vertex_idx;

    let ray = rays[index];
    let last_lens = system.interfaces[system.interface_count - 1];

    let scale = 1 / last_lens.sa_half;

    var out_pos = ray.pos;

    let vertex_coords = vec2u(grid_local_vertex_idx % GS, grid_local_vertex_idx / GS);
    out.tex_coords = vec2f(vertex_coords) / f32(GS - 1);

    out.clip_position = camera.view_proj * vec4f(out_pos.xy * scale, 0, 1);
    out.position = out_pos;

    out.tex = ray.tex;

    let wavelength = WAVELENGTHS[in.instance_index % 3];

    var col = wavelength_to_rgb(wavelength);
    out.color = col;

    let barycentric_coords = array(
        vec3f(1, 0, 0),
        vec3f(0, 1, 0),
        vec3f(0, 0, 1),
    );

    out.barycentric_coord = barycentric_coords[in.vertex_index % 3];

    out.area_ratio = calculate_grid_cell_area_ratio(index);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    if in.tex.z > 1 {
        discard;
    }

//    let focus = 25.0;
//    let aperture_diameter = 14.0;
//    let f = aperture_diameter / focus;
//
//    let f_stop = f / pow(sqrt(2.0), 5);

    let cutoff = smoothstep(1.0, 0.9, in.tex.z);
    let alpha = in.tex.a;

    let tex = textureSample(ghost_texture, ghost_sampler, in.tex_coords);
    var tex_intensity = saturate(length(in.tex_coords - 0.5) * 0.5);
	tex_intensity = smoothstep(0., 1., tex_intensity);
	tex_intensity = mix(0.5, 1., tex_intensity);

    let vignette = smoothstep(1.0, 0.9, length(in.tex.xy) - 0.9);

    let area_ratio = in.area_ratio;

    let intensity = tex * tex_intensity * alpha * vignette * area_ratio * params.intensity * cutoff;

    if all(intensity == vec4f(0)) {
        discard;
    }

    let color = vec4f(in.color.xyz, 1);

    let final_color = intensity * color;

    if params.debug_wireframe_alpha <= 0 {
        return final_color;
    }

    let edge = edge_factor(in.barycentric_coord);

//    let edge_color = vec4f(in.tex_coords, 0, 1) * 0.0001;
    let edge_color = vec4f(1) * params.debug_wireframe_alpha;

    return mix(final_color, edge_color, 1-edge);
}
