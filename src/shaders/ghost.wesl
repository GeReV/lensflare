struct VertexOutput {
  @builtin(position) position : vec4f,
}

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3f,
}

@vertex
fn vs_main(in : VertexInput) -> VertexOutput {
  var output: VertexOutput;
  output.position = vec4(in.position, 1.0);
  return output;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  return vec4f(1);
}