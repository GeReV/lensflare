@group(0) @binding(0) var tex_sampler : sampler;
@group(0) @binding(1) var texture : texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
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

  var output: VertexOutput;
  output.position = vec4(pos[vertex_index], 0.0, 1.0);
  output.uv = uv[vertex_index];
  return output;
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  var color = textureSample(texture, tex_sampler, uv);

  color = vec4f(color.xyz, 1);

  return color;
}