use std::convert::TryFrom;
use itertools::Itertools;
use sscanf::sscanf;

#[derive(Debug, sscanf::FromScanf)]
#[sscanf(format_unescaped = "{radius}\t{axis_position}\t{n}\t{aperture}", )]
pub struct LensInterface {
    pub radius: f32,
    pub axis_position: f32,
    pub n: f32,
    pub aperture: f32,
}

impl TryFrom<&str> for LensInterface {
    type Error = sscanf::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        sscanf!(value, "{LensInterface}")
    }
}

pub fn parse_lenses(
    data: &str,
) -> Result<Vec<LensInterface>, <LensInterface as TryFrom<&str>>::Error> {
    data.lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(&str::trim)
        .map(LensInterface::try_from)
        .map_ok(|lens| LensInterface {
            radius: lens.radius,
            axis_position: lens.axis_position,
            n: lens.n,
            aperture: lens.aperture,
        })
        .collect()
}
