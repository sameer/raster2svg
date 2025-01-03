use ndarray::prelude::*;
use ndarray::Array3;
use std::fmt;
use std::fmt::Display;
use std::num::ParseIntError;
use std::ops::Index;
use std::str::FromStr;

use crate::ColorModel;

impl ColorModel {
    pub fn convert(&self, rgb: ArrayView3<f64>) -> Array3<f64> {
        match self {
            ColorModel::Cielab => ciexyz_to_cielab(srgb_to_ciexyz(rgb).view()),
            ColorModel::Rgb => rgb.to_owned(),
        }
    }
    pub fn convert_single(&self, color: &Color) -> [f64; 3] {
        let rgb = a_to_nd(color.as_ref());
        nd_to_a::<3>(match self {
            ColorModel::Cielab => ciexyz_to_cielab(srgb_to_ciexyz(rgb.view()).view()),
            ColorModel::Rgb => rgb,
        })
    }

    pub fn cylindrical(&self, image: ArrayView3<f64>) -> Array3<f64> {
        match self {
            ColorModel::Cielab => cielab_to_ciehcl(image),
            ColorModel::Rgb => srgb_to_hsl(image),
        }
    }
    pub fn cylindrical_single(&self, color: [f64; 3]) -> [f64; 3] {
        let color = a_to_nd(&color);
        nd_to_a(match self {
            ColorModel::Cielab => cielab_to_ciehcl(color.view()),
            ColorModel::Rgb => srgb_to_hsl(color.view()),
        })
    }

    /// CIEHCL: based on CIEDE2000
    pub fn cylindrical_diff(&self, reference: [f64; 3], actual: [f64; 3]) -> f64 {
        match self {
            ColorModel::Cielab => {
                fn g(cmid: f64) -> f64 {
                    0.5 * (1. - (cmid.powi(7) / (cmid.powi(7) + 25.0_f64.powi(7))).sqrt())
                }
                let [h1, c1, l1] = reference;
                let (b1, a1) = {
                    let (a, b) = h1.sin_cos();
                    (a * c1, b * c1)
                };
                let [h2, c2, l2] = actual;
                let (b2, a2) = {
                    let (a, b) = h2.sin_cos();
                    (a * c2, b * c2)
                };

                let δ_lprime = l2 - l1;
                let lmid = (l1 + l2) / 2.;
                let cmid = (c1 + c2) / 2.;

                let a1_prime = a1 * (1. + g(cmid));
                let a2_prime = a2 * (1. + g(cmid));
                let c1_prime = (a1_prime.powi(2) + b1.powi(2)).sqrt();
                let c2_prime = (a2_prime.powi(2) + b2.powi(2)).sqrt();
                let cmid_prime = (c1_prime + c2_prime) / 2.;
                let δ_cprime = c2_prime - c1_prime;

                let h1_prime = (b1.atan2(a1_prime).to_degrees() + 360.) % 360.;
                let h2_prime = (b2.atan2(a2_prime).to_degrees() + 360.) % 360.;
                let raw_δ_hprime = (h1_prime - h2_prime).abs();
                let δ_hprime_precursor = if raw_δ_hprime <= 180. {
                    h2_prime - h1_prime
                } else if h2_prime <= h1_prime {
                    h2_prime - h1_prime + 360.
                } else {
                    h2_prime - h1_prime - 360.
                };
                let δ_hprime = 2.0
                    * (c1_prime * c2_prime).sqrt()
                    * (δ_hprime_precursor / 2.).to_radians().sin();
                let hmid_prime = if raw_δ_hprime <= 180. {
                    (h1_prime + h2_prime) / 2.
                } else if h1_prime + h2_prime < 360. {
                    (h1_prime + h2_prime + 360.) / 2.
                } else {
                    (h1_prime + h2_prime - 360.) / 2.
                };

                let t = 1. - 0.17 * (hmid_prime - 30.).to_radians().cos()
                    + 0.24 * (2. * hmid_prime).to_radians().cos()
                    + 0.32 * (3. * hmid_prime + 6.).to_radians().cos()
                    - 0.20 * (4. * hmid_prime - 63.).to_radians().cos();
                let sl = 1. + (0.015 * (lmid - 50.).powi(2)) / (20. + (lmid - 50.).powi(2)).sqrt();
                let sc = 1. + 0.045 * cmid_prime;
                let sh = 1. + 0.015 * cmid_prime * t;

                let dtheta = 30. * (-((hmid_prime - 275.) / 25.).powi(2)).exp();
                let rc = 2. * (cmid_prime.powi(7) / (cmid_prime.powi(7) + 25.0_f64.powi(7))).sqrt();
                let rt = -(2.0 * dtheta).to_radians().sin() * rc;

                let kl = 1.;
                let kc = 1.;
                let kh = 1.;

                ((δ_lprime / (kl * sl)).powi(2)
                    + (δ_cprime / (kc * sc)).powi(2)
                    + (δ_hprime / (kh * sh)).powi(2)
                    + rt * δ_cprime / (kc * sc) * δ_hprime / (kh * sh))
                    .sqrt()
            }
            ColorModel::Rgb => todo!(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Color([f64; 3]);

impl Index<usize> for Color {
    type Output = f64;
    fn index(&'_ self, i: usize) -> &'_ Self::Output {
        &self.0[i]
    }
}

impl From<[f64; 3]> for Color {
    fn from(color: [f64; 3]) -> Self {
        Self(color)
    }
}

impl AsRef<[f64; 3]> for Color {
    fn as_ref(&self) -> &[f64; 3] {
        &self.0
    }
}

impl From<Color> for [u8; 3] {
    fn from(c: Color) -> Self {
        [
            (c[0] * 255.).round() as u8,
            (c[1] * 255.).round() as u8,
            (c[2] * 255.).round() as u8,
        ]
    }
}

impl Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "#{:02x}{:02x}{:02x}",
            (self[0] * 255.).round() as u8,
            (self[1] * 255.).round() as u8,
            (self[2] * 255.).round() as u8
        )
    }
}

pub enum ColorParseError {
    Int(ParseIntError),
    Length(usize),
    MissingPound,
}

impl fmt::Display for ColorParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(e) => write!(f, "{}", e),
            Self::Length(l) => write!(f, "Unexpected length {} should be 3 or 6", l),
            Self::MissingPound => write!(f, "Color should be preceded by a pound symbol"),
        }
    }
}

impl From<ParseIntError> for ColorParseError {
    fn from(e: ParseIntError) -> Self {
        Self::Int(e)
    }
}

impl FromStr for Color {
    type Err = ColorParseError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        if let Some(hex) = input.strip_prefix('#') {
            let parsed = u32::from_str_radix(hex, 16)?;
            let mut res = [0.; 3];
            match hex.len() {
                3 => {
                    for (i, res_i) in res.iter_mut().enumerate() {
                        // Hex shorthand: convert 0xFFF into 1.0, 1.0, 1.0
                        let digit = (parsed >> (8 - 4 * i) & 0xF) as u8;
                        *res_i = (digit << 4 | digit) as f64 / 255.;
                    }
                }
                6 => {
                    for (i, res_i) in res.iter_mut().enumerate() {
                        *res_i = ((parsed >> (16 - 8 * i) & 0xFF) as u8) as f64 / 255.;
                    }
                }
                other => return Err(ColorParseError::Length(other)),
            }
            Ok(Self(res))
        } else {
            Err(ColorParseError::MissingPound)
        }
    }
}

pub fn a_to_nd<const N: usize>(x: &[f64; N]) -> Array3<f64> {
    Array3::<f64>::from_shape_vec((N, 1, 1), x.as_ref().to_vec()).unwrap()
}

pub fn nd_to_a<const N: usize>(a: Array3<f64>) -> [f64; N] {
    let view = a.slice(s![.., 0, 0]);
    let mut res = [0.; N];
    for i in 0..N {
        res[i] = view[i];
    }
    res
}

/// sRGB to Hue, Saturation, Lightness (HSL)
///
/// <https://en.wikipedia.org/wiki/HSL_and_HSV#From_RGB>
pub fn srgb_to_hsl(srgb: ArrayView3<f64>) -> Array3<f64> {
    let mut hsl = srgb.to_owned();
    hsl.slice_mut(s![0, .., ..])
        .assign(&srgb.map_axis(Axis(0), |rgb| {
            let v = rgb[0].max(rgb[1]).max(rgb[2]);
            let c = v - rgb[0].min(rgb[1]).min(rgb[2]);
            if c <= f64::EPSILON {
                0.
            } else if (v - rgb[0]).abs() <= f64::EPSILON {
                std::f64::consts::FRAC_PI_3 * (0. + (rgb[1] - rgb[2]) / c)
            } else if (v - rgb[1]).abs() <= f64::EPSILON {
                std::f64::consts::FRAC_PI_3 * (2. + (rgb[2] - rgb[0]) / c)
            } else {
                std::f64::consts::FRAC_PI_3 * (4. + (rgb[0] - rgb[1]) / c)
            }
        }));

    hsl.slice_mut(s![1, .., ..])
        .assign(&srgb.map_axis(Axis(0), |rgb| {
            let v = rgb[0].max(rgb[1]).max(rgb[2]);
            let c = v - rgb[0].min(rgb[1]).min(rgb[2]);
            let l = v - c / 2.;

            if l.abs() <= f64::EPSILON || (l - 1.).abs() <= f64::EPSILON {
                0.
            } else {
                (v - l) / (l.min(1. - l))
            }
        }));

    hsl.slice_mut(s![2, .., ..])
        .assign(&srgb.map_axis(Axis(0), |rgb| {
            let v = rgb[0].max(rgb[1]).max(rgb[2]);
            let c = v - rgb[0].min(rgb[1]).min(rgb[2]);
            v - c / 2.
        }));

    hsl
}

/// CAT02 D65 -> D50 Illuminant transform
///
/// Derived using [colour-science](https://colour.readthedocs.io/en/develop/index.html):
/// ```python
/// from colour import CCS_ILLUMINANTS
/// from colour.adaptation import matrix_chromatic_adaptation_VonKries
/// from colour.models import xy_to_xyY, xyY_to_XYZ
///
/// illuminant_RGB = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
/// illuminant_XYZ = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
/// chromatic_adaptation_transform = "CAT02"
/// print(
///     matrix_chromatic_adaptation_VonKries(
///         xyY_to_XYZ(xy_to_xyY(illuminant_RGB)),
///         xyY_to_XYZ(xy_to_xyY(illuminant_XYZ)),
///         transform=chromatic_adaptation_transform,
///     )
/// )
/// ```
const CHROMATIC_ADAPTATION_TRANSFORM: [[f64; 3]; 3] = [
    [1.04257389, 0.03089108, -0.05281257],
    [0.02219345, 1.00185663, -0.02107375],
    [-0.00116488, -0.00342053, 0.76178908],
];

/// sRGB under D65 illuminant to CIEXYZ under D50 illuminant
///
/// <https://en.wikipedia.org/wiki/SRGB#The_reverse_transformation_(sRGB_to_CIE_XYZ)>
pub fn srgb_to_ciexyz(srgb: ArrayView3<f64>) -> Array3<f64> {
    let mut ciexyz = Array3::<f64>::zeros(srgb.raw_dim());

    const SRGB_TO_CIEXYZ: [[f64; 3]; 3] = [
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
    ];

    for i in 0..ciexyz.raw_dim()[1] {
        for j in 0..ciexyz.raw_dim()[2] {
            let mut ciexyz_under_d65 = [0.; 3];
            for k in 0..3 {
                let gamma_expanded = gamma_expand_rgb(srgb[[k, i, j]]);
                for (l, ciexyz_under_d65_l) in ciexyz_under_d65.iter_mut().enumerate() {
                    *ciexyz_under_d65_l += SRGB_TO_CIEXYZ[l][k] * gamma_expanded;
                }
            }
            for ciexyz_under_d65_l in ciexyz_under_d65.iter_mut() {
                *ciexyz_under_d65_l = ciexyz_under_d65_l.clamp(0., 1.);
            }

            for k in 0..3 {
                for (l, ciexyz_under_d65_l) in IntoIterator::into_iter(ciexyz_under_d65).enumerate()
                {
                    ciexyz[[k, i, j]] += CHROMATIC_ADAPTATION_TRANSFORM[k][l] * ciexyz_under_d65_l;
                }
                ciexyz[[k, i, j]] = ciexyz[[k, i, j]].clamp(0., 1.);
            }
        }
    }

    ciexyz
}

/// CIEXYZ to CIELAB, both under D50 illuminant
///
/// <https://en.wikipedia.org/wiki/CIELAB_color_space#From_CIEXYZ_to_CIELAB>
pub fn ciexyz_to_cielab(ciexyz: ArrayView3<f64>) -> Array3<f64> {
    // Can't find my source for these, but one derivation is on https://www.mathworks.com/help/images/ref/whitepoint.html
    const X_N: f64 = 0.96429568;
    const Y_N: f64 = 1.;
    const Z_N: f64 = 0.8251046;
    let mut cielab = Array3::<f64>::zeros(ciexyz.raw_dim());
    cielab
        .slice_mut(s![0, .., ..])
        .assign(&ciexyz.map_axis(Axis(0), |xyz| {
            let y = xyz[1];
            let l = 116. * cielab_f(y / Y_N) - 16.;
            l.clamp(0., 100.)
        }));

    cielab
        .slice_mut(s![1, .., ..])
        .assign(&ciexyz.map_axis(Axis(0), |xyz| {
            let x = xyz[0];
            let y = xyz[1];
            500. * (cielab_f(x / X_N) - cielab_f(y / Y_N))
        }));

    cielab
        .slice_mut(s![2, .., ..])
        .assign(&ciexyz.map_axis(Axis(0), |xyz| {
            let y = xyz[1];
            let z = xyz[2];
            200. * (cielab_f(y / Y_N) - cielab_f(z / Z_N))
        }));

    cielab
}

/// CIELAB to its cylindrical equivalent CIEHCL
///
/// This is usually LCH or HLC, but it is made HCL here to
/// align with HSL
pub fn cielab_to_ciehcl(cielab: ArrayView3<f64>) -> Array3<f64> {
    let mut ciehcl = Array3::<f64>::zeros(cielab.raw_dim());

    // L remains the same
    ciehcl
        .slice_mut(s![2, .., ..])
        .assign(&cielab.slice(s![0, .., ..]));

    // Euclidean distance from origin to define chromaticity
    ciehcl
        .slice_mut(s![1, .., ..])
        .assign(&cielab.map_axis(Axis(0), |lab| (lab[1].powi(2) + lab[2].powi(2)).sqrt()));

    // Hue angle
    ciehcl
        .slice_mut(s![0, .., ..])
        .assign(&cielab.map_axis(Axis(0), |lab| lab[2].atan2(lab[1])));

    ciehcl
}

/// Function defined in CIEXYZ to CIELAB conversion
///
/// <https://en.wikipedia.org/wiki/CIELAB_color_space#From_CIEXYZ_to_CIELAB>
fn cielab_f(t: f64) -> f64 {
    const DELTA_POW3: f64 = 216. / 24389.;
    const THREE_DELTA_POW2: f64 = 108. / 841.;
    const FOUR_OVER_TWENTY_NINE: f64 = 4. / 29.;

    if t > DELTA_POW3 {
        t.cbrt()
    } else {
        t / THREE_DELTA_POW2 + FOUR_OVER_TWENTY_NINE
    }
}

/// Gamma-expand (or linearize) an sRGB value
///
/// <https://en.wikipedia.org/wiki/SRGB#From_sRGB_to_CIE_XYZ>
fn gamma_expand_rgb(component: f64) -> f64 {
    if component <= 0.04045 {
        component / 12.92
    } else {
        ((component + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_to_hsl_is_correct() {
        let arr = array![[
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 1., 1.],
            [0.89, 0.89, 0.89],
            [0., 0., 0.],
        ]]
        .reversed_axes();
        assert_eq!(
            srgb_to_hsl(arr.view()).reversed_axes(),
            array![[
                [0., 1., 0.5],
                [std::f64::consts::FRAC_PI_3 * 2., 1., 0.5],
                [std::f64::consts::FRAC_PI_3 * 4., 1., 0.5],
                [0., 0., 1.],
                [0., 0., 0.89],
                [0., 0., 0.],
            ]]
        );
    }

    #[test]
    fn srgb_to_ciexyz_is_correct() {
        let arr = array![[
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 1., 1.],
            [0., 0., 0.]
        ]]
        .reversed_axes();
        assert_eq!(
            srgb_to_ciexyz(arr.view()).reversed_axes(),
            array![[
                [0.43550563324299996, 0.221740574943, 0.013494928054000002],
                [0.3886224651359999, 0.7219522484959999, 0.08794233419200001],
                [0.14021657533599996, 0.05630936703600001, 0.723623297434],
                [0.969044992445, 1.0, 0.75726133156],
                [0.0, 0.0, 0.0]
            ]]
        );
    }

    #[test]
    fn ciexyz_to_cielab_is_correct() {
        let arr = array![[
            [0.43550563324299996, 0.221740574943, 0.013494928054000002],
            [0.3886224651359999, 0.7219522484959999, 0.08794233419200001],
            [0.14021657533599996, 0.05630936703600001, 0.723623297434],
            [0.969044992445, 1.0, 0.75726133156],
            [0.0, 0.0, 0.0]
        ]]
        .reversed_axes();
        assert_eq!(
            ciexyz_to_cielab(arr.view()).reversed_axes(),
            array![[
                [54.21119728870907, 80.98253915088577, 70.28651365985937],
                [88.06247407807324, -79.21970052415361, 84.5923008745313],
                [28.461577872330834, 71.28097404863831, -114.78144041108523],
                [100.0, 0.8195163861430821, 5.639091770210247],
                [0.0, 0.0, 0.0]
            ]]
        );
    }

    #[test]
    fn test_ciede2000() {
        let color_model = ColorModel::Cielab;

        let test_data = vec![
            [
                1., 50., 2.6772, -79.7751, 50., 0., -82.7485, 79.82, 82.7485, 271.9222, 270.,
                270.9611, 0.0001, 0.6907, 1., 4.6578, 1.8421, -1.7042, 2.0425,
            ],
            [
                2., 50., 3.1571, -77.2803, 50., 0., -82.7485, 77.3448, 82.7485, 272.3395, 270.,
                271.1698, 0.0001, 0.6843, 1., 4.6021, 1.8216, -1.707, 2.8615,
            ],
            [
                3., 50., 2.8361, -74.02, 50., 0., -82.7485, 74.0743, 82.7485, 272.1944, 270.,
                271.0972, 0.0001, 0.6865, 1., 4.5285, 1.8074, -1.706, 3.4412,
            ],
            [
                4., 50., -1.3802, -84.2814, 50., 0., -82.7485, 84.2927, 82.7485, 269.0618, 270.,
                269.5309, 0.0001, 0.7357, 1., 4.7584, 1.9217, -1.6809, 1.,
            ],
            [
                5., 50., -1.1848, -84.8006, 50., 0., -82.7485, 84.8089, 82.7485, 269.1995, 270.,
                269.5997, 0.0001, 0.7335, 1., 4.77, 1.9218, -1.6822, 1.,
            ],
            [
                6., 50., -0.9009, -85.5211, 50., 0., -82.7485, 85.5258, 82.7485, 269.3964, 270.,
                269.6982, 0.0001, 0.7303, 1., 4.7862, 1.9217, -1.684, 1.,
            ],
            [
                7., 50., 0., 0., 50., -1., 2., 0., 2.5, 0., 126.8697, 126.8697, 0.5, 1.22, 1.,
                1.0562, 1.0229, 0., 2.3669,
            ],
            [
                8., 50., -1., 2., 50., 0., 0., 2.5, 0., 126.8697, 0., 126.8697, 0.5, 1.22, 1.,
                1.0562, 1.0229, 0., 2.3669,
            ],
            [
                9., 50., 2.49, -0.001, 50., -2.49, 0.0009, 3.7346, 3.7346, 359.9847, 179.9862,
                269.9854, 0.4998, 0.7212, 1., 1.1681, 1.0404, -0.0022, 7.1792,
            ],
            [
                10., 50., 2.49, -0.001, 50., -2.49, 0.001, 3.7346, 3.7346, 359.9847, 179.9847,
                269.9847, 0.4998, 0.7212, 1., 1.1681, 1.0404, -0.0022, 7.1792,
            ],
            [
                11., 50., 2.49, -0.001, 50., -2.49, 0.0011, 3.7346, 3.7346, 359.9847, 179.9831,
                89.9839, 0.4998, 0.6175, 1., 1.1681, 1.0346, 0., 7.2195,
            ],
            [
                12., 50., 2.49, -0.001, 50., -2.49, 0.0012, 3.7346, 3.7346, 359.9847, 179.9816,
                89.9831, 0.4998, 0.6175, 1., 1.1681, 1.0346, 0., 7.2195,
            ],
            [
                13., 50., -0.001, 2.49, 50., 0.0009, -2.49, 2.49, 2.49, 90.0345, 270.0311,
                180.0328, 0.4998, 0.9779, 1., 1.1121, 1.0365, 0., 4.8045,
            ],
            [
                14., 50., -0.001, 2.49, 50., 0.001, -2.49, 2.49, 2.49, 90.0345, 270.0345, 180.0345,
                0.4998, 0.9779, 1., 1.1121, 1.0365, 0., 4.8045,
            ],
            [
                15., 50., -0.001, 2.49, 50., 0.0011, -2.49, 2.49, 2.49, 90.0345, 270.038, 0.0362,
                0.4998, 1.3197, 1., 1.1121, 1.0493, 0., 4.7461,
            ],
            [
                16., 50., 2.5, 0., 50., 0., -2.5, 3.7496, 2.5, 0., 270., 315., 0.4998, 0.8454, 1.,
                1.1406, 1.0396, -0.0001, 4.3065,
            ],
            [
                17., 50., 2.5, 0., 73., 25., -18., 3.4569, 38.9743, 0., 332.4939, 346.247, 0.3827,
                1.4453, 1.1608, 1.9547, 1.4599, -0.0003, 27.1492,
            ],
            [
                18., 50., 2.5, 0., 61., -5., 29., 3.4954, 29.8307, 0., 103.5532, 51.7766, 0.3981,
                0.6447, 1.064, 1.7498, 1.1612, 0., 22.8977,
            ],
            [
                19., 50., 2.5, 0., 56., -27., -3., 3.5514, 38.4728, 0., 184.4723, 272.2362, 0.4206,
                0.6521, 1.0251, 1.9455, 1.2055, -0.8219, 31.903,
            ],
            [
                20., 50., 2.5, 0., 58., 24., 15., 3.5244, 37.0102, 0., 23.9095, 11.9548, 0.4098,
                1.1031, 1.04, 1.912, 1.3353, 0., 19.4535,
            ],
            [
                21., 50., 2.5, 0., 50., 3.1736, 0.5854, 3.7494, 4.7954, 0., 7.0113, 3.5056, 0.4997,
                1.2616, 1., 1.1923, 1.0808, 0., 1.,
            ],
            [
                22., 50., 2.5, 0., 50., 3.2972, 0., 3.7493, 4.945, 0., 0., 0., 0.4997, 1.3202, 1.,
                1.1956, 1.0861, 0., 1.,
            ],
            [
                23., 50., 2.5, 0., 50., 1.8634, 0.5757, 3.7497, 2.8536, 0., 11.638, 5.819, 0.4999,
                1.2197, 1., 1.1486, 1.0604, 0., 1.,
            ],
            [
                24., 50., 2.5, 0., 50., 3.2592, 0.335, 3.7493, 4.8994, 0., 3.9206, 1.9603, 0.4997,
                1.2883, 1., 1.1946, 1.0836, 0., 1.,
            ],
            [
                25., 60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 49.759, 52.2238,
                133.2085, 130.9584, 132.0835, 0.0017, 1.301, 1.1427, 3.2946, 1.9951, 0., 1.2644,
            ],
            [
                26., 63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 33.1427, 31.5202,
                190.1951, 187.449, 188.8221, 0.049, 0.9402, 1.1831, 2.4549, 1.456, 0., 1.263,
            ],
            [
                27., 61.2901, 3.7196, -5.3901, 61.4292, 2.248, -4.962, 7.7487, 5.995, 315.924,
                304.1385, 310.0313, 0.4966, 0.6952, 1.1586, 1.3092, 1.0717, -0.0032, 1.8731,
            ],
            [
                28., 35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 44.5557, 40.355,
                175.1161, 177.7418, 176.429, 0.0063, 1.0168, 1.2148, 2.9105, 1.6476, 0., 1.8645,
            ],
            [
                29., 22.7233, 20.0904, -46.694, 23.0331, 14.973, -42.5619, 50.8532, 45.1317,
                293.3339, 289.4279, 291.3809, 0.0026, 0.3636, 1.4014, 3.1597, 1.2617, -1.2537,
                2.0373,
            ],
            [
                30., 36.4612, 47.858, 18.3852, 36.2715, 50.5065, 21.2231, 51.3256, 54.8444,
                20.9901, 22.766, 21.8781, 0.0013, 0.9239, 1.1943, 3.3888, 1.7357, 0., 1.4146,
            ],
            [
                31., 90.8027, -2.0831, 1.441, 91.1528, -1.6435, 0.0447, 3.4408, 2.4655, 155.241,
                178.9612, 167.1011, 0.4999, 1.1546, 1.611, 1.1329, 1.0511, 0., 1.4441,
            ],
            [
                32., 90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.227, 1.5298, 228.6315,
                208.2412, 218.4363, 0.5, 1.3916, 1.593, 1.062, 1.0288, 0., 1.5381,
            ],
            [
                33., 6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 2.4636, 2.2335, 259.8025,
                266.2073, 263.0049, 0.4999, 0.9556, 1.6517, 1.1057, 1.0337, -0.0004, 0.6377,
            ],
            [
                34., 2.0776, 0.0795, -1.135, 0.9033, -0.0636, -0.5514, 1.1412, 0.5596, 275.9978,
                260.1842, 268.091, 0.5, 0.7826, 1.7246, 1.0383, 1.01, 0., 0.9082,
            ],
        ];
        for [pair, l1, a1, b1, l2, a2, b2, c1, c2, h1, h2, h_avg, g, t, sl, sc, sh, rt, expected] in
            test_data
        {
            let hcl_a = color_model.cylindrical_single([l1, a1, b1]);
            let hcl_b = color_model.cylindrical_single([l2, a2, b2]);
            let epsilon = 1e-4;
            let actual = color_model.cylindrical_diff(hcl_a, hcl_b);
            assert!(
                (expected - actual).abs() < epsilon,
                "{expected} {actual} {pair} {c1} {c2} {h1} {h2} {h_avg} {g} {t} {sl} {sc} {sh} {rt}",
            );
        }
    }
}
