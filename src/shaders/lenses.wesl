import package::utils::PI;
import package::lenses_types::{Lens, LensInterface, Ray, Intersection, LensSystem};

@group(1) @binding(0)
var<storage, read> lenses: array<Lens>;

@group(1) @binding(1)
var<uniform> system: LensSystem;

// First two elements are the bounces, element 3 is length.
// Merged to handle 16-byte memory alignment.
@group(1) @binding(2)
var<storage, read> bounces_and_lengths: array<vec3u>;

fn compute_ref_index(lambda_in_micrometer: f32, lens: Lens) -> f32 {
        let lambda_pow2 = lambda_in_micrometer * lambda_in_micrometer;
        let lambda_pow4 = lambda_pow2 * lambda_pow2;
        let inv_lambda_pow2 = 1 / lambda_pow2;
        let inv_lambda_pow4 = inv_lambda_pow2 * inv_lambda_pow2;
        let inv_lambda_pow6 = inv_lambda_pow4 * inv_lambda_pow2;
        let inv_lambda_pow8 = inv_lambda_pow6 * inv_lambda_pow2;
        let inv_lambda_pow10 = inv_lambda_pow8 * inv_lambda_pow2;
        let inv_lambda_pow12 = inv_lambda_pow10 * inv_lambda_pow2;
        
        return sqrt(lens.a0
		+ lens.a1 * lambda_pow2
		+ lens.a2 * lambda_pow4
		+ lens.a3 * inv_lambda_pow2
		+ lens.a4 * inv_lambda_pow4
		+ lens.a5 * inv_lambda_pow6
		+ lens.a6 * inv_lambda_pow8
		+ lens.a7 * inv_lambda_pow10
		+ lens.a8 * inv_lambda_pow12);
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
    let sin_theta0 = sin(theta0);
    let theta1 = asin(sin_theta0 * n0 / n1);
    let theta2 = asin(sin_theta0 * n0 / n2);

    // amplitude for outer refl./transmission on topmost interface
    let theta0plus1 = theta0 + theta1;
    let theta0minus1 = theta0 - theta1;
    let rs01 = -sin(theta0minus1) / sin(theta0plus1);
    let rp01 = tan(theta0minus1) / tan(theta0plus1);
    let ts01 = 2.0 * sin(theta1) * cos(theta0) / sin(theta0plus1);
    let tp01 = ts01 * cos(theta0minus1);

    // amplitude for inner reflection
    let theta1plus2 = theta1 + theta2;
    let theta1minus2 = theta1 - theta2;
    let rs12 = -sin ( theta1minus2 ) / sin ( theta1plus2 ) ;
    let rp12 = tan ( theta1minus2 ) / tan ( theta1plus2 ) ;

    // after passing through first surface twice:
    // 2 transmissions and 1 reflection
    let ris = ts01*ts01*rs12 ;
    let rip = tp01*tp01*rp12 ;

    // phase difference between outer and inner reflections
    let dy = d1 * n1 ;
    let dx = tan ( theta1 ) * dy;
    let delay = sqrt ( dx*dx+dy*dy ) ;
    let col_rel_phase = cos(4 * PI / lambda * (delay - dx*sin_theta0));

    // Add up sines of different phase and amplitude
    let out_s2 = rs01*rs01 + ris*ris + 2*rs01*ris * col_rel_phase;
    let out_p2 = rp01*rp01 + rip*rip + 2*rp01*rip * col_rel_phase;

    // reflectivity
    return ( out_s2 + out_p2 ) * 0.5;
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
    const EPSILON: f32 = PI * 0.0001;

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
        i.norm *= -1;
    }

    i.theta = acos ( dot(-r.dir , i.norm ) ) ;
    //avoid getting hole on grid
    if (abs(i.theta) < EPSILON)
    {
        i.theta += EPSILON;
    }
    i.hit = true;
    i.inverted = t < 0; // mark an inverted ray

    return i;
}

fn cull_ray(ray: ptr<function, Ray>) {
    ray.pos = vec3f(0, 0, -1);
    ray.tex.a = 0;
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
            cull_ray(&r);
           break;
       }

       // record texture coord. or max. rel. radius
       if f.flat_surface == 0 {
        r.tex.z = max(r.tex.z, length(i.pos.xy) / f.sa_half);
       } else if t == system.aperture_index {
        // iris aperture plane
        let v = i.pos.xy / system.interfaces[system.aperture_index].sa_half;
        r.tex.x = v.x;
        r.tex.y = v.y;
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
       var n0_idx: u32;
       var n2_idx: u32;


        if r.dir.z < 0 {
            n0_idx = f.n.x;
            n2_idx = f.n.y;
        } else {
            n0_idx = f.n.y;
            n2_idx = f.n.x;
        }

        var n0: f32;
        var n2: f32;

        if n0_idx >= 0 {
            n0 = compute_ref_index(lambda * 1e-3, lenses[n0_idx]);
        } else {
            n0 = 1;
        }

        if n2_idx >= 0 {
            n2 = compute_ref_index(lambda * 1e-3, lenses[n2_idx]);
        } else {
            n2 = 1;
        }

       let n1 = max(sqrt(n0 * n2), 1.38);

       if !bReflect // refraction
       {
           const EPSILON = 0.00001;

           r.dir = refract ( r.dir , i.norm , n0 / n2 ) ;

           if length(r.dir) < EPSILON {
            // total reflection
//            r.tex.a = 3.0;
                cull_ray(&r);
                break;
            }
       }
       else
       {
           // reflection with anti-reflection coating
           r.dir = reflect ( r.dir , i.norm ) ;
           var R = fresnel_anti_reflect(i.theta + 0.0001, lambda, f.d1, n0, n1, n2);

            // TODO: Not sure why NaNs occur here.
           R = max(0.0, R);

           r.tex.a *= R; // update ray intensity
       }

       t += delta;
   }

   if k<length {
    // early-exit rays = invalid
    cull_ray(&r);
   }

   return r;
}