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

fn luminance(v: vec3f) -> f32 {
    return dot(v, vec3f(0.2126, 0.7152, 0.0722));
}

fn reinhard(v: vec3f) -> vec3f {
    let l = luminance(v);

    let factor = l / (l + 1);

    return v / (1 + v);
}

fn reinhard2(x: vec3f) -> vec3f {
  const L_white = 4.0;

  let l = luminance(x);

  let factor = (l * (1.0 + l / (L_white * L_white))) / (1.0 + l);

  return x * factor;
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  var color = textureSample(texture, tex_sampler, uv);

  color = vec4f(color.xyz, 1);

  return color;
}