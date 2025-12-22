import package::utils::{PI, rotation_mat2};
import constants::{base_size, min_size};

@group(0) @binding(0) var tex_sampler : sampler;
@group(0) @binding(1) var texture : texture_2d<f32>;

@group(1) @binding(0) var dust_tex_sampler : sampler;
@group(1) @binding(1) var dust_texture : texture_2d<f32>;

@group(2) @binding(0)
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
  var out = vec4f();

  let d = length(input.uv * 2 - 1);

  let uv = input.uv - 0.5;

  let clamped = f32(d < 1);

  let count = arrayLength(&colors);
  for(var i: u32 = 0; i < count; i++) {
    let t = f32(i) / f32(count - 1);
    let scale1 = mix(0.7, 1, t);
    let scale2 = mix(1.0, 1.5, t);

    let scaled_uv1 = (uv / scale1) + 0.5;
    let scaled_uv2 = (uv / scale2) + 0.5;
//    let uv = input.uv;

    var p = textureSample(texture, tex_sampler, scaled_uv1);
    var l = length(p.xy);
    let starburst_val = l * l;

    p = textureSample(dust_texture, dust_tex_sampler, scaled_uv2);
    l = length(p.xy);
    let dust_val = l * l;

    let fade = mix(1.0, 0, d) * clamped;

    let color_mix = mix(vec4f(1), colors[i], 0.8);

    let starburst = (starburst_val * d * (1 - t) * color_mix) * 0.0001;
    let dust = pow(dust_val * smoothstep(0, 1, 0.5 - d) * colors[i], vec4f(1.1)) * 0.0002;

    out += (starburst + dust) * fade;
  }

  out /= f32(count);

  return out;
}
