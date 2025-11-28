use glam::{Mat3, Vec3};

/**
 * Convert a wavelength in the visible light spectrum to a RGB color value that is suitable to be displayed on a
 * monitor
 *
 * @param wavelength wavelength in nm
 * @return RGB color encoded in int. each color is represented with 8 bits and has a layout of
 * 00000000RRRRRRRRGGGGGGGGBBBBBBBB where MSB is at the leftmost
 */
pub fn wavelength_to_rgb(wavelength: f32) -> Vec3 {
    let xyz = cie1931_wavelength_to_xyz_fit(wavelength);

    srgb_xyz_to_rgb(xyz)
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
fn srgb_xyz_to_rgb(xyz: Vec3) -> Vec3 {
    let m = Mat3::from_cols_array(&[
        3.2406255, -0.9689307, 0.0557101, -1.537208, 1.8757561, -0.2040211, -0.4986286, 0.0415175,
        1.0569959,
    ]);

    srgb_xyz_to_rgb_postprocess(m * xyz)
}

/**
 * helper function for {@link #srgbXYZ2RGB(double[])}
 */
fn srgb_xyz_to_rgb_postprocess(rgbl: Vec3) -> Vec3 {
    // clip if c is out of range
    let result = rgbl.clamp(Vec3::ZERO, Vec3::ONE);

    // apply the color component transfer function
    result.map(|c| {
        if c <= 0.0031308 {
            12.92 * c
        } else {
            1.055 * c.powf(2.4) - 0.055
        }
    })
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
fn cie1931_wavelength_to_xyz_fit(wavelength: f32) -> Vec3 {
    let wave = Vec3::splat(wavelength);

    let mut xyz = Vec3::ZERO;

    {
        let c = Vec3::new(442.0, 599.8, 501.1);
        let t = (wave - c)
            * Vec3::select(
                wave.cmplt(c),
                Vec3::new(0.0624, 0.0264, 0.0490),
                Vec3::new(0.0374, 0.0323, 0.0382),
            );

        xyz.x = Vec3::new(0.362, 1.056, -0.065).dot(Vec3::exp(-0.5 * t * t));
    }

    {
        let c = Vec3::new(568.8, 530.9, 0.0);
        let t = (wave - c)
            * Vec3::select(
                wave.cmplt(c),
                Vec3::new(0.0213, 0.613, 0.0),
                Vec3::new(0.0247, 0.0322, 0.0),
            );

        xyz.y = Vec3::new(0.821, 0.286, 0.0).dot(Vec3::exp(-0.5 * t * t));
    }

    {
        let c = Vec3::new(437.0, 459.0, 0.0);
        let t = (wave - c)
            * Vec3::select(
                wave.cmplt(c),
                Vec3::new(0.0845, 0.0385, 0.0),
                Vec3::new(0.0278, 0.0725, 0.0),
            );

        xyz.z = Vec3::new(1.217, 0.681, 0.0).dot(Vec3::exp(-0.5 * t * t));
    }

    xyz
}
