import constants::{base_size, min_size};

@group(0) @binding(0) var tex_sampler : sampler;
@group(0) @binding(1) var texture : texture_2d<f32>;

@group(1) @binding(0)
var<storage, read> colors: array<vec4f>;

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
  @location(1) color: vec4f,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index : u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
  const pos = array(
    vec2( -1.0,  3.0),
    vec2( 3.0, -1.0),
    vec2(-1.0, -1.0),
  );

  const uv = array(
    vec2(0.0, 2.0),
    vec2(2.0, 0.0),
    vec2(0.0, 0.0),
  );

  let color_count = arrayLength(&colors);

  let scale = mix(min_size, 1, f32(instance_index) / f32(color_count - 1)) * f32(base_size);
  let uv_scale = 1 / scale;

  var output: VertexOutput;

  let out_uv = ((uv[vertex_index] - 0.5) * uv_scale) + 0.5;

  output.position = vec4(pos[vertex_index], 0.0, 1.0);
  output.uv = out_uv;
  output.color = colors[instance_index];

  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  var p = textureSample(texture, tex_sampler, input.uv);

  return vec4f(p.r) * input.color;
}