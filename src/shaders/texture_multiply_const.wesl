import package::complex::Complex;
import constants::{workgroup_size_x, workgroup_size_y, workgroup_size_z, C};

@if (rg32)
@group(0) @binding(0) var src: texture_storage_2d<rg32float, read>;
@else
@group(0) @binding(0) var src: texture_storage_2d<rgba32float, read>;

@if (rg32)
@group(0) @binding(1) var dst: texture_storage_2d<rg32float, write>;
@else
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;

@compute
@workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)
fn process(@builtin(global_invocation_id) gid: vec3u) {
    // Perform dst = src * C
    let p = textureLoad(src, gid.xy);

    textureStore(dst, gid.xy, p * C);
}