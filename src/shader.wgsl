struct Camera {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}
@group(0) @binding(0)
var<uniform> camera: Camera;

struct LensInterface {
    center: vec3f, // center of sphere / plane on z-axis
    n: vec3f, // refractive indices (n0, n1, n2)
    radius: f32, // radius of sphere/plane
    sa: f32, // nominal radius (from optical axis)
    d1: f32, // coating thickness = lambdaAR / 4 / n1
    flat_surface: u32, // is this interface a plane?
}

struct Ray {
    pos: vec3f,
    dir: vec3f,
    tex: vec4f,
}

struct Intersection {
    pos: vec3f,
    norm: vec3f,
    theta: f32, // incident angle
    hit: bool, // intersection found?
    inverted: bool, // inverted intersection?
}

const NUM_INTERFACES: u32 = 32;
const NUM_BOUNCES: u32 = NUM_INTERFACES * (NUM_INTERFACES - 1) / 2;

struct LensSystem {
    interfaces: array<LensInterface, NUM_INTERFACES>,
    interface_count: u32,
    bounce_count: u32,
    aperture_index: i32,
}

struct GridLimits {
    tl: vec3f,
    br: vec3f,
}

struct Params {
    ray_dir: vec3f,
    bid: i32,
    intensity: f32,
    lambda: f32,
    wireframe: u32,
}

@group(1) @binding(0)
var<uniform> system: LensSystem;

// First two elements are the bounces, element 3 is length.
// Merged to handle 16-byte memory alignment.
@group(1) @binding(1)
var<storage, read> bounces_and_lengths: array<vec3u, NUM_BOUNCES>;

@group(1) @binding(2)
var<storage, read> grid_limits: array<GridLimits, NUM_BOUNCES>;

@group(2) @binding(0)
var<uniform> params: Params;

const PI = radians(180.0);

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

fn fresnel_anti_reflect(
    theta0: f32, // angle of incidence
    lambda: f32, // wavelength of ray
    d1: f32, // thickness of anti-reflection coating
    n0: f32, // RI (refr. index) of 1st medium
    n1: f32, // RI of coating layer
    n2: f32, // RI of the 2nd medium
) -> f32 {
    // refraction angle sin coating and the 2nd medium
    let theta1 = asin(sin(theta0) * n0 / n1);
    let theta2 = asin(sin(theta0) * n0 / n2);

    // amplitude for outer refl./transmission on topmost interface
    let rs01 = -sin(theta0 - theta1) / sin(theta0 + theta1);
    let rp01 = tan(theta0 - theta1) / tan(theta0 + theta1);
    let ts01 = 2.0 * sin(theta1) * cos(theta0) / sin(theta0 + theta1);
    let tp01 = ts01 * cos(theta0 - theta1);

    // amplitude for inner reflection
    let rs12 = -sin ( theta1-theta2 ) / sin ( theta1+theta2 ) ;
    let rp12 = tan ( theta1-theta2 ) / tan ( theta1+theta2 ) ;

    // after passing through first surface twice:
    // 2 transmissions and 1 reflection
    let ris = ts01*ts01*rs12 ;
    let rip = tp01*tp01*rp12 ;

    // phase difference between outer and inner reflections
    let dy = d1*n1 ;
    let dx = tan ( theta1 ) *dy ;
    let delay = sqrt ( dx*dx+dy*dy ) ;
    let rel_phase = 4 * PI / lambda * (delay - dx*sin(theta0));

    // Add up sines of different phase and amplitude
    let out_s2 = rs01*rs01 + ris*ris + 2*rs01*ris*cos ( rel_phase ) ;
    let out_p2 = rp01*rp01 + rip*rip + 2*rp01*rip*cos ( rel_phase ) ;

    return ( out_s2+out_p2 ) / 2 ; // reflectivity
}

fn testFLAT(r: Ray, f: LensInterface) -> Intersection {
    var i: Intersection;
    i.pos = r.pos + r.dir * ( ( f.center.z - r.pos.z ) / r.dir.z ) ;
    i.theta = 0 ; // meaningless
    i.hit = true ;
    i.inverted = false ;

    if r.dir.z > 0 {
        i.norm = vec3f(0, 0, -1);
    } else {
        i.norm = vec3f(0, 0, 1);
    }

    return i;
}

fn testSPHERE(r: Ray, f: LensInterface) -> Intersection {
    var i: Intersection;

    let d = r.pos - f.center;
    let b = dot ( d , r.dir ) ;
    let c = dot ( d , d ) - f.radius * f.radius ;
    let b2_c = b*b - c ;

    if b2_c < 0 {
        // no intersection
        i.hit = false;
        return i;
    }

    var sgn: f32;
    if f.radius * r.dir.z > 0 {
        sgn = 1.0;
    } else {
        sgn = -1.0;
    }
    let t = sqrt ( b2_c ) * sgn - b ;

    i.pos = r.dir * t + r.pos;
    i.norm = normalize( i.pos - f.center);

    if dot(i.norm, r.dir) > 0 {
        i.norm = -i.norm;
    }

    i.theta = acos ( dot(-r.dir , i.norm ) ) ;
    i.hit = true;
    i.inverted = t < 0 ; // mark an inverted ray

    return i;
}

fn trace(
   bid: i32 , // index of current bounce/ghost
   r_in: Ray, // input ray from the entrance plane
   lambda: f32, // wavelength of ray
) -> Ray {
   var r: Ray = r_in;

   let surfaces = bounces_and_lengths[bid].xy; // read 2 surfaces to reflect
   let length = bounces_and_lengths[bid].z; // length of intersections

   // initialization
   var phase: u32 = 0; // ray-tracing phase
   var delta: i32 = 1; // delta for for-loop
   var t: i32 = 1; // index of target in tr face to test

   var k: u32 = 0;
   for (; k < length; k++) {
       let f = system.interfaces[t];
       let bReflect = u32(t) == surfaces[phase];
       if bReflect {
        delta = -delta;
        phase++;
       }

       // i n t e r s e c t i o n t e s t
       var i: Intersection;

       if f.flat_surface == 1 {
        i = testFLAT ( r , f );
        } else {
         i = testSPHERE ( r , f );
       }

       if !i.hit {
           // exit upon miss
//           r.tex.a = 2.0;
           break;
       }

       // record texture coord. or max. rel. radius
       if f.flat_surface == 0 {
        r.tex.z = max(r.tex.z, length(i.pos.xy) / f.sa);
       } else if t == system.aperture_index { // iris aperture plane
        r.tex.x = i.pos.x / system.interfaces[system.aperture_index].sa;
        r.tex.y = i.pos.y / system.interfaces[system.aperture_index].sa;
       }

       // update ray direction and position
       r.dir = normalize( i.pos - r.pos );

       if i.inverted {
            r.dir *= -1.0; // corrected inverted ray
       }

       r.pos = i.pos ;

       // skip reflection/refraction for flat surfaces
       if f.flat_surface == 1 {
        t += delta;
        continue;
       }
       // do reflection/refraction for spher. surfaces
       var n0: f32;
       var n1 = f.n.y;
       var n2: f32;

        if r.dir.z < 0 {
            n0 = f.n.x;
            n2 = f.n.z;
        } else {
            n0 = f.n.z;
            n2 = f.n.x;
        }

//        let lambda_offset = (1 - ((lambda - 380) / (800 - 380))) * 0.005;
//
//        n1 += lambda_offset;
//
//        if n0 > 1 {
//            n0 += lambda_offset;
//        } else {
//            n2 += lambda_offset;
//        }

       if !bReflect // refraction
       {
           r.dir = refract ( r.dir , i.norm , n0 / n2 ) ;
           if all(r.dir == vec3f(0)) {
            // total reflection
//            r.tex.a = 3.0;
           break ;
            }
       }
       else // reflection with anti-reflection coating
       {
           r.dir = reflect ( r.dir , i.norm ) ;
           let R = fresnel_anti_reflect(i.theta, lambda, f.d1, n0, n1, n2) ;
           r.tex.a *= R ; // update ray intensity
       }

       t += delta;
   }

   if k<length {
    // early-exit rays = invalid
        r.tex.a = 0;
//        r.tex.a = f32(length - k);
   }
//    r.tex.a = f32(k) / f32(length);

   return r;
}

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3f,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) position: vec3f,
    @location(1) color: vec4f,
    @location(2) tex: vec4f,
//    @location(1) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;

//    let light_pos = vec3f(-1.0, -1.0, 8.0);
//    let ray_dir = normalize(pos - light_pos);

    let lambda = in.instance_index % 3;

    var bid = i32(in.instance_index / 3);
    if params.bid >= 0 {
        bid = params.bid;
    }

    let grid_limit = grid_limits[bid];

    var in_pos: vec3f;
    in_pos = mix(grid_limit.tl, grid_limit.br, in.position);

    // Set lens entrance center to the center of the lens with an offset relative to the ray's position and the lens' radius,
    // i.e., radius * 0.5 away from center of the lens.
    let ray_pos = system.interfaces[0].center + in_pos;
    let ray_dir = normalize(params.ray_dir);

//    let light_pos = pos * system.interfaces[0].sa;
//    let ray_dir = vec3f(0, 0, 1);
    let ray_tex = vec4f(in.position.xy, 1.0, params.intensity);

    let wavelengths = array<f32, 3>(645, 520, 408);

    let wavelength = wavelengths[lambda];

    let in_ray = Ray(ray_pos, ray_dir, ray_tex);
    let out_ray = trace(bid, in_ray, wavelength);

    var col = wavelength_to_rgb(wavelength);

//    col = vec3f();
//    switch (lambda) {
//        case 0: {
//          col.x = 1;
//        }
//        case 1: {
//          col.y = 1;
//        }
//        case 2: {
//          col.z = 1;
//        }
//        default: { }
//    }

    out.position = in.position;
    out.tex = out_ray.tex;

    var pos = out_ray.pos;
//    if out.tex.z > 1 {
//        pos = in_pos;
//    }

    out.clip_position = camera.view_proj * vec4f(pos.xy, 0, 1);
//    out.clip_position = camera.view_proj * vec4f(pos, 1.0);

//    col = vec3f(in.position.xy, 0);
//    if out.tex.z > 1 {
//        col = vec3f(0.1);
//    }

    out.color = vec4f(col, 1);


//    out.color = vec4f(f32(bounces_and_lengths[params.bid].x) / 11.0, f32(bounces_and_lengths[params.bid].y) / 11.0, 0.0, 1.0);
//    out.color = vec4f(hsv2rgb(vec3f(f32(bid) / 38.0, 0.5, 0.5)), 0.1);

//    out.color = vec4f(hsv2rgb(vec3f(out_ray.tex.a / 6, 1, 1)), 0.1);

//    out.color = vec4f(hsv2rgb(vec3f(out_ray.tex.a, 1, 1)), 0.1);

//    out.color = vec3f(1.0);

    return out;
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
  const L_white = 16.0;

  return (x * (1.0 + x / (L_white * L_white))) / (1.0 + x);
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {

    let aperture = smoothstep(0.4, 0.3, length(in.tex.xy));

    var alpha = in.tex.a;
    if min(alpha, 1000) == 1000 {
        alpha = 0;
    }

    var color = vec4f(in.color.xyz, 1) * aperture * alpha;

    if in.tex.z > 1 {
        color.a = 0;
    }
//    if params.wireframe == 1 {
//        color.a = max(0.05, color.a);
//    }

//    if in.tex.z > 1 {
//        color.a = 0;
//    }

    return color;
}
