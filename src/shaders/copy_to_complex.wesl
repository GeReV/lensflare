import package::complex::Complex;
import constants::{workgroup_size_x, workgroup_size_y, workgroup_size_z};

@group(0)
@binding(0)
var src: texture_2d<f32>;

@group(0)
@binding(1)
var dst: texture_storage_2d<rg32float, write>;

@compute
@workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)
fn copy(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) num_workgroups: vec3u) {
    let p = textureLoad(src, gid.xy, 0);

    textureStore(dst, gid.xy, vec4f(p.x, 0, 0, 0));
}