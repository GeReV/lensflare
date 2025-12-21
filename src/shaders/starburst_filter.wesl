import package::utils::{PI, rotation_mat2};
import constants::{base_size, min_size};

@group(0) @binding(0) var tex_sampler : sampler;
@group(0) @binding(1) var texture : texture_2d<f32>;

@group(1) @binding(0)
var<storage, read> colors: array<vec4f>;

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
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

  var output: VertexOutput;

  let out_uv = uv[vertex_index];

  output.position = vec4(pos[vertex_index], 0.0, 1.0);
  output.uv = out_uv;

  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
  // Spiral filter as used by https://www.jpgrenier.org/lensflare.html
  // This basically just copies the texture multiple times in a small outwards spiral,
  // creating a blur effect.
  var out = vec4f();

  let d = length(input.uv * 2 - 1);

  let uv = input.uv - 0.5;

  const rotation = 0.03 * PI;

  let count = arrayLength(&colors);
  for(var i: u32 = 0; i < count; i++) {
    let t = f32(i) / f32(count - 1);
    let a = 2 * PI * t * 2;

    let spiral = vec2f(cos(a), sin(a)) * 0.002 * t;
    let rot = rotation_mat2(t * 0.05 + rotation);
    let rotated_uv = rot * (uv + spiral);

    let starburst = textureSample(texture, tex_sampler, rotated_uv + 0.5);

    out += starburst;
  }

  out /= f32(count);

//  return textureSample(texture, tex_sampler, input.uv);

  return out;
}