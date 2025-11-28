import constants;
import package::utils::PI;

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
}

// signed distance to a regular n-gon
fn sd_ngon(pos: vec2f, r: f32) -> f32 {
    var p = pos;

    // these 2 lines can be precomputed
    let an = 2 * PI / constants::polygon_sides;
    let he = r * tan(0.5 * an);

    // rotate to first sector
    p = -p.yx; // if you want the corner to be up

    let a_sin = sin(-constants::polygon_rotation);
    let a_cos = cos(-constants::polygon_rotation);
    p = mat2x2f(a_cos, -a_sin, a_sin, a_cos) * p;

    let bn = an * floor((atan2(p.y, p.x) + 0.5 * an) / an);
    let cs = vec2f(cos(bn), sin(bn));
    p = mat2x2f(cs.x, -cs.y, cs.y, cs.x) * p;

    // side of polygon
    return length(p - vec2f(r, clamp(p.y, -he, he))) * sign(p.x - r);
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
    let p = uv * 2 - 1;
    let a = sd_ngon(p, constants::polygon_radius);

    // Inside border
//    return vec4f(1 - smoothstep(-0.02, 0.0, a));

    // Outside border
    return vec4f(1 - smoothstep(0.02, 0.025, a));
}