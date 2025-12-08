import package::complex::Complex;
import constants::{workgroup_size_x, workgroup_size_y, workgroup_size_z};

@group(0)
@binding(0)
var a: texture_storage_2d<rgba32float, read>;

@group(0)
@binding(1)
var b: texture_storage_2d<r32float, read>;

@group(0)
@binding(2)
var dst: texture_storage_2d<rgba32float, write>;

@group(0)
@binding(3)
var<storage, read> color: vec4f;

@compute
@workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)
fn process(@builtin(global_invocation_id) gid: vec3u) {
    // Perform dst = a + b * color
    let pa = textureLoad(a, gid.xy);
    let pb = vec4f(textureLoad(b, gid.xy).x);

    textureStore(dst, gid.xy, pa + pb * color);
}