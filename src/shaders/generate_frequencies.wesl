import package::complex::{Complex, complex_magnitude, complex_exp, complex_mul, complex_mul_scalar};
import constants::{workgroup_size_x, workgroup_size_y, workgroup_size_z, size};
import package::utils::PI;

struct Parameters {
    delta: f32,
    z: f32,
    wavelength_meters: f32,
}

@group(0)
@binding(0)
var dst: texture_storage_2d<rg32float, write>;

@group(0)
@binding(1)
var<storage, read> params: Parameters;

@compute
@workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)
fn generate(@builtin(global_invocation_id) gid: vec3u) {
    let k = 2.0 * PI / params.wavelength_meters;

    let s = f32(size / 2);

    let fxy = (vec2f(gid.xy) - s) / (params.delta * s);
    let kxy = fxy * 2.0 * PI;

    let kz2 = k * k - (kxy.x * kxy.x + kxy.y * kxy.y);

    var kz: Complex;

    if kz2 >= 0 {
        kz = Complex(sqrt(kz2), 0);
    } else {
        kz = Complex(0, -sqrt(-kz2));
    }

    let i = Complex(0, 1);
    let c = complex_exp(complex_mul_scalar(complex_mul(i, kz), params.z));


    textureStore(dst, gid.xy, vec4f(c.re, c.im, 0, 0));
}