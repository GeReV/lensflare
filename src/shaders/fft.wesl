import package::complex::{Complex, complex_add, complex_mul, complex_mul_scalar, complex_conjugate, complex_magnitude, complex_normalize, complex_cis};
import constants::{workgroup_size_x, workgroup_size_y, workgroup_size_z, size};
import package::utils::PI;

@group(0)
@binding(0)
var src: texture_storage_2d<rg32float, read>;

@group(0)
@binding(1)
var dst: texture_storage_2d<rg32float, write>;

var<workgroup> row_a: array<Complex, size>;
var<workgroup> row_b: array<Complex, size>;

fn get_coord(gid: vec3u) -> vec2u {
    @if (fft_cols)
    return gid.xy;
    @else
    return gid.yx;
}

fn fft_step_stockham(a: Complex, b: Complex, theta: f32) -> Complex {
    let e = complex_cis(-PI * 2 * theta);

    return complex_add(a, complex_mul(b, e));
}

@compute
@workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)
fn fft(@builtin(global_invocation_id) gid: vec3u) {
    let n = u32(log2(size));

    let x = get_coord(gid);

    let c = textureLoad(src, x).xy;
    row_a[gid.x] = Complex(c.x, c.y);
    workgroupBarrier();

    for (var k: u32 = 0; k < n; k++) {
        let n_s = u32(2 << k);

        let theta = f32(gid.x) / f32(n_s);

        let base = gid.x / n_s * (n_s / 2);
        let offset = gid.x % (n_s / 2);

        let idx0 = base + offset;
        let idx1 = idx0 + size / 2;

        if k % 2 == 0 {
            let a = row_a[idx0];
            let b = row_a[idx1];

            row_b[gid.x] = fft_step_stockham(a, b, theta);
        } else {
            let a = row_b[idx0];
            let b = row_b[idx1];

            row_a[gid.x] = fft_step_stockham(a, b, theta);
        }

        workgroupBarrier();
    }

    var temp: Complex;
    if n % 2 == 0 {
        temp = row_a[gid.x];
    } else {
        temp = row_b[gid.x];
    }

    textureStore(dst, x, vec4f(temp.re, temp.im, 0.0, 0.0));
}

@compute
@workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)
fn ifft(@builtin(global_invocation_id) gid: vec3u) {
    let n = u32(log2(size));

    let x = get_coord(gid);

    let c = textureLoad(src, x);

    row_a[gid.x] = complex_conjugate(Complex(c.x, c.y));

    workgroupBarrier();

    for (var k: u32 = 0; k < n; k++) {
        let n_s = u32(2 << k);

        let theta = f32(gid.x) / f32(n_s);

        let base = gid.x / n_s * (n_s / 2);
        let offset = gid.x % (n_s / 2);

        let idx0 = base + offset;
        let idx1 = idx0 + size / 2;

        if k % 2 == 0 {
            let a = row_a[idx0];
            let b = row_a[idx1];

            row_b[gid.x] = fft_step_stockham(a, b, theta);
        } else {
            let a = row_b[idx0];
            let b = row_b[idx1];

            row_a[gid.x] = fft_step_stockham(a, b, theta);
        }

        workgroupBarrier();
    }

    var temp: Complex;
    if n % 2 == 0 {
        temp = row_a[gid.x];
    } else {
        temp = row_b[gid.x];
    }

    temp = complex_conjugate(temp);

    textureStore(dst, x, vec4f(temp.re, temp.im, 0, 0));
}

@compute
@workgroup_size(workgroup_size_x / 2, workgroup_size_y, workgroup_size_z)
fn fftshift(@builtin(global_invocation_id) gid: vec3u) {
    let idx_a = get_coord(gid);
    let idx_b = get_coord(gid + vec3u(size / 2, 0, 0));

    let a = textureLoad(src, idx_a);
    let b = textureLoad(src, idx_b);

    textureStore(dst, idx_b, a);
    textureStore(dst, idx_a, b);
}