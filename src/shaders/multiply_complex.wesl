import package::complex::{Complex, complex_mul};
import constants::{workgroup_size_x, workgroup_size_y, workgroup_size_z, size};

@group(0)
@binding(0)
var src: texture_storage_2d<rg32float, read>;

@group(0)
@binding(1)
var dst: texture_storage_2d<rg32float, write>;

@group(0)
@binding(2)
var coefficients: texture_storage_2d<rg32float, read>;

@compute
@workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)
fn multiply(@builtin(global_invocation_id) gid: vec3u) {
    let x = gid.y * size + gid.x;

    let p = textureLoad(src, gid.xy);

    let coef_p = textureLoad(coefficients, gid.xy);
    let coef = Complex(coef_p.x, coef_p.y);

    var c = Complex(p.x, p.y);

    c = complex_mul(c, coef);

    textureStore(dst, gid.xy, vec4f(c.re, c.im, 0, 0));
}