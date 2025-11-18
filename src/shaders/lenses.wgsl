#define_import_path lenses

#import utils::{PI}

const NUM_INTERFACES: u32 = 32;
const NUM_BOUNCES: u32 = NUM_INTERFACES * (NUM_INTERFACES - 1) / 2;

struct LensInterface {
    center: vec3f, // center of sphere / plane on z-axis
    n: vec3f, // refractive indices (n0, n1, n2)
    radius: f32, // radius of sphere/plane
    sa: f32, // nominal radius (from optical axis)
    d: f32, // coating thickness = lambdaAR / 4 / n1
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

struct LensSystem {
    interfaces: array<LensInterface, NUM_INTERFACES>,
    interface_count: u32,
    bounce_count: u32,
    aperture_index: i32,
}

@group(1) @binding(0)
var<uniform> system: LensSystem;

// First two elements are the bounces, element 3 is length.
// Merged to handle 16-byte memory alignment.
@group(1) @binding(1)
var<storage, read> bounces_and_lengths: array<vec3u, NUM_BOUNCES>;

@group(1) @binding(2)
var<storage, read> vertices: array<vec3f>;

@group(1) @binding(3)
var<storage, read> indices: array<u32>;

fn fresnel_anti_reflect(
    theta0: f32, // angle of incidence
    lambda: f32, // wavelength of ray
    d: f32, // thickness of anti-reflection coating
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
    let dy = d*n1 ;
    let dx = tan ( theta1 ) *dy ;
    let delay = sqrt ( dx*dx+dy*dy ) ;
    let rel_phase = 4 * PI / lambda * (delay - dx*sin(theta0));

    // Add up sines of different phase and amplitude
    let out_s2 = rs01*rs01 + ris*ris + 2*rs01*ris*cos ( rel_phase ) ;
    let out_p2 = rp01*rp01 + rip*rip + 2*rp01*rip*cos ( rel_phase ) ;

    return ( out_s2+out_p2 ) / 2 ; // reflectivity
}

fn test_flat(r: Ray, f: LensInterface) -> Intersection {
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

fn test_sphere(r: Ray, f: LensInterface) -> Intersection {
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
        i = test_flat ( r , f );
        } else {
         i = test_sphere ( r , f );
       }

       if !i.hit {
           // exit upon miss
//           r.tex.a = 2.0;
           break;
       }

       // record texture coord. or max. rel. radius
       if f.flat_surface == 0 {
        r.tex.z = max(r.tex.z, length(i.pos.xy) / f.sa);
       } else if t == system.aperture_index {
        // iris aperture plane
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
           let R = fresnel_anti_reflect(i.theta, lambda, f.d, n0, n1, n2) ;
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