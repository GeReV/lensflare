/**
 * Convert a wavelength in the visible light spectrum to a RGB color value that is suitable to be displayed on a
 * monitor
 *
 * @param wavelength wavelength in nm
 * @return RGB color encoded in int. each color is represented with 8 bits and has a layout of
 * 00000000RRRRRRRRGGGGGGGGBBBBBBBB where MSB is at the leftmost
 */
fn wavelength_to_rgb(wavelength: f32) -> vec3f {
    let xyz = cie1931_wavelength_to_xyz_fit(wavelength);

    return srgb_xyz_to_rgb(xyz);
}

/**
 * Convert XYZ to RGB in the sRGB color space
 * <p>
 * The conversion matrix and color component transfer function is taken from http://www.color.org/srgb.pdf, which
 * follows the International Electrotechnical Commission standard IEC 61966-2-1 "Multimedia systems and equipment -
 * Colour measurement and management - Part 2-1: Colour management - Default RGB colour space - sRGB"
 *
 * @param xyz XYZ values in a double array in the order of X, Y, Z. each value in the range of [0.0, 1.0]
 * @return RGB values in a double array, in the order of R, G, b. each value in the range of [0.0, 1.0]
 */
fn srgb_xyz_to_rgb(xyz: vec3f) -> vec3f {
    let M = mat3x3(
        3.2406255, -0.9689307, 0.0557101,
        -1.537208 , 1.8757561, -0.2040211,
        -0.4986286, 0.0415175, 1.0569959,
    );

    return srgb_xyz_to_rgb_postprocess(M * xyz);
}

/**
 * helper function for {@link #srgbXYZ2RGB(double[])}
 */
fn srgb_xyz_to_rgb_postprocess(rgbl: vec3f) -> vec3f {
    // clip if c is out of range
    let result = clamp(rgbl, vec3f(0), vec3f(1));

    // apply the color component transfer function
    return select(result * 12.92, 1.055 * pow(result, vec3f(1. / 2.4)) - 0.055, result <= vec3f(0.0031308));
}

/**
 * A multi-lobe, piecewise Gaussian fit of CIE 1931 XYZ Color Matching Functions by Wyman el al. from Nvidia. The
 * code here is adopted from the Listing 1 of the paper authored by Wyman et al.
 * <p>
 * Reference: Chris Wyman, Peter-Pike Sloan, and Peter Shirley, Simple Analytic Approximations to the CIE XYZ Color
 * Matching Functions, Journal of Computer Graphics Techniques (JCGT), vol. 2, no. 2, 1-11, 2013.
 *
 * @param wavelength wavelength in nm
 * @return XYZ in a double array in the order of X, Y, Z. each value in the range of [0.0, 1.0]
 */
fn cie1931_wavelength_to_xyz_fit(wavelength: f32) -> vec3f {
    let wave = vec3f(wavelength);

    var xyz = vec3f();

    {
        let c = vec3f(442.0, 599.8, 501.1);
        let t = (wave - c) * select(vec3f(0.0624, 0.0264, 0.0490), vec3f(0.0374, 0.0323, 0.0382), wave < c);


        xyz.x = dot(vec3f(0.362, 1.056, -0.065), exp(-0.5 * t * t));
    }

    {
        let c = vec3f(568.8, 530.9, 0.0);
        let t = (wave - c) * select(vec3f(0.0213, 0.613, 0), vec3f(0.0247, 0.0322, 0), wave < c);

        xyz.y = dot(vec3f(0.821, 0.286, 0), exp(-0.5 * t * t));
    }

    {
        let c = vec3f(437.0, 459.0, 0);
        let t = (wave - c) * select(vec3f(0.0845, 0.0385, 0), vec3f(0.0278, 0.0725, 0), wave < c);

        xyz.z = dot(vec3f(1.217, 0.681, 0), exp(-0.5 * t * t));
    }

    return xyz;
}

fn rgb2hsv(c: vec3f) -> vec3f
{
    let K = vec4f(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = mix(vec4f(c.bg, K.wz), vec4f(c.gb, K.xy), step(c.b, c.g));
    let q = mix(vec4f(p.xyw, c.r), vec4f(c.r, p.yzx), step(p.x, c.r));

    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn hsv2rgb(c: vec3f) -> vec3f
{
    let K = vec4f(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = vec3f(abs(fract(c.xxx + K.xyz) * 6.0 - K.www));
    return c.z * mix(K.xxx, clamp(vec3f(p) - K.xxx, vec3f(0.0), vec3f(1.0)), c.y);
}

fn luminance(v: vec3f) -> f32 {
    return dot(v, vec3f(0.2126, 0.7152, 0.0722));
}

fn reinhard(v: vec3f) -> vec3f {
    let l = luminance(v);

    let factor = l / (l + 1);

    return v / (1 + v);
}

fn reinhard2(x: vec3f) -> vec3f {
  const L_white = 4.0;

  let l = luminance(x);

  let factor = (l * (1.0 + l / (L_white * L_white))) / (1.0 + l);

  return x * factor;
}