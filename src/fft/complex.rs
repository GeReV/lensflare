use glam::Vec2;

#[derive(Default, Debug, Copy, Clone, PartialEq)]
pub(crate) struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
    pub const I: Complex = Complex { re: 0.0, im: 1.0 };

    pub const fn new(re: f32, im: f32) -> Self {
        Complex { re, im }
    }

    #[inline(always)]
    pub fn re(&self) -> f32 {
        self.re
    }

    #[inline(always)]
    pub fn im(&self) -> f32 {
        self.im
    }

    pub fn exp(angle: f32) -> Self {
        Complex::new(angle.cos(), angle.sin())
    }

    pub fn conjugate(&self) -> Self {
        Complex {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn magnitude(&self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn normalize(&mut self) {
        let magnitude = self.magnitude();
        self.re /= magnitude;
        self.im /= magnitude;
    }

    pub fn scale(&mut self, s: f32) {
        self.re *= s;
        self.im *= s;
    }
}

impl std::ops::Add<Complex> for Complex {
    type Output = Complex;

    fn add(self, rhs: Complex) -> Self::Output {
        Complex {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Sub<Complex> for Complex {
    type Output = Complex;

    fn sub(self, rhs: Complex) -> Self::Output {
        self + -rhs
    }
}

impl std::ops::Mul<Complex> for Complex {
    type Output = Complex;

    fn mul(self, rhs: Complex) -> Self::Output {
        Complex {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Neg for Complex {
    type Output = Complex;

    fn neg(self) -> Self::Output {
        Complex {
            re: -self.re,
            im: -self.im,
        }
    }
}

#[cfg(test)]
mod tests {}
