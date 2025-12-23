import super::utils::{PI, Params};
import super::camera::camera;
import super::lenses::system;

@group(2) @binding(0)
var<uniform> params: Params;

@group(3) @binding(0) var tex_sampler : sampler;
@group(3) @binding(1) var texture : texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
  const pos = array(
    vec2( -1.0,  1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0, -1.0),
    vec2( -1.0,  1.0),
    vec2( 1.0, 1.0),
    vec2( 1.0, -1.0),
  );

  const uv = array(
    vec2(0.0, 1.0),
    vec2(1.0, 0.0),
    vec2(0.0, 0.0),
    vec2(0.0, 1.0),
    vec2(1.0, 1.0),
    vec2(1.0, 0.0),
  );

  let first_lens = system.interfaces[0];
  let last_lens = system.interfaces[system.interface_count - 1];

  let ray_origin = first_lens.center;
  let ray_dir = normalize(params.ray_dir);

  let axis = last_lens.center - ray_origin;
  let t = dot(ray_dir, axis);

  let ray_hit = ray_origin + ray_dir * t;

  let vert_pos = ray_hit + vec3f(pos[vertex_index], 0) * params.starburst_scale;
  let out_pos = camera.view_proj * vec4(vert_pos.xy / vert_pos.z, 0, 1);

  var output: VertexOutput;

  output.position = out_pos;
  output.uv = uv[vertex_index];

  return output;
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  var color = textureSample(texture, tex_sampler, uv);

  color = vec4f(color.xyz, 1) * params.intensity;

  return color;
}