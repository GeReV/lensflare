use glam::Vec2;

pub struct Arc {
    center: Vec2,
    radius: f32,
    start: f32,
    end: f32,
}

impl Arc {
    pub fn circle(center: Vec2, radius: f32) -> Self {
        Self::new(center, radius, 0.0, 2.0 * std::f32::consts::PI)
    }

    pub fn new(center: Vec2, radius: f32, start: f32, end: f32) -> Self {
        Self {
            center,
            radius,
            start,
            end,
        }
    }

    pub fn iter(&self, count: usize) -> impl Iterator<Item = Vec2> {
        ArcIter::new(&self, count)
    }
}

pub struct ArcIter<'a> {
    arc: &'a Arc,
    count: usize,
    current: usize,
}

impl<'a> ArcIter<'a> {
    pub fn new(arc: &'a Arc, count: usize) -> Self {
        Self {
            arc,
            count,
            current: 0,
        }
    }
}

impl<'a> Iterator for ArcIter<'a> {
    type Item = Vec2;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current > self.count {
            return None;
        }

        let delta = (self.arc.end - self.arc.start) / self.count as f32;
        let current_angle = self.arc.start + delta * self.current as f32;

        let result = Vec2::new(
            self.arc.center.x + self.arc.radius * current_angle.cos(),
            self.arc.center.y + self.arc.radius * current_angle.sin(),
        );

        self.current += 1;

        Some(result)
    }
}
