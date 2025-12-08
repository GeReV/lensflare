import package::complex::{Complex, complex_magnitude};
import constants::{workgroup_size_x, workgroup_size_y, workgroup_size_z};

@group(0)
@binding(0)
var src: texture_storage_2d<rg32float, read>;

@group(0)
@binding(1)
var dst: texture_storage_2d<r32float, write>;

@compute
@workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)
fn normalize(@builtin(global_invocation_id) gid: vec3u) {
    let p = textureLoad(src, gid.xy);

    let c = Complex(p.x, p.y);

    textureStore(dst, gid.xy, vec4f(complex_magnitude(c), 0, 0, 0));
}