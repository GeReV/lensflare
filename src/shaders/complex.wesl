struct Complex {
	re: f32,
	im: f32,
}

fn complex_add(a: Complex, b: Complex) -> Complex {
	return Complex(
	    a.re + b.re,
        a.im + b.im
    );
}

fn complex_mul(a: Complex, b: Complex) -> Complex {
	return Complex(
	    a.re * b.re - a.im * b.im,
        a.re * b.im + a.im * b.re
    );
}

fn complex_mul_scalar(a: Complex, b: f32) -> Complex {
    return Complex(a.re * b, a.im * b);
}

fn complex_conjugate(a: Complex) -> Complex {
	return Complex(a.re, -a.im);
}

fn complex_magnitude_sq(a: Complex) -> f32 {
	return a.re * a.re + a.im * a.im;
}

fn complex_magnitude(a: Complex) -> f32 {
	return sqrt(complex_magnitude_sq(a));
}

fn complex_normalize(a: Complex) -> Complex {
	let mag = complex_magnitude(a);
	return Complex(a.re / mag, a.im / mag);
}

fn complex_cis(theta: f32) -> Complex {
	return Complex(cos(theta), sin(theta));
}

fn complex_exp(a: Complex) -> Complex {
	return Complex(exp(a.re) * cos(a.im), exp(a.re) * sin(a.im));
}