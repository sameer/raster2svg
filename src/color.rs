use log::info;
use ndarray::prelude::*;
use ndarray::Array3;
use std::fmt;
use std::fmt::Display;
use std::num::ParseIntError;
use std::ops::Index;
use std::str::FromStr;

#[derive(Clone, Copy, Debug)]
pub struct Color([f64; 3]);

impl Index<usize> for Color {
    type Output = f64;
    fn index(&'_ self, i: usize) -> &'_ Self::Output {
        &self.0[i]
    }
}

impl AsRef<[f64; 3]> for Color {
    fn as_ref(&self) -> &[f64; 3] {
        &self.0
    }
}

impl Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "#{:02x}{:02x}{:02x}",
            (self[0] * 256.) as u8,
            (self[1] * 256.) as u8,
            (self[2] * 256.) as u8
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
                    for i in 0..3 {
                        // Hex shorthand: convert 0xFFF into 1.0, 1.0, 1.0
                        let digit = (parsed >> (8 - 4 * i) & 0xF) as u8;
                        res[i] = (digit << 4 | digit) as f64 / 255.;
                    }
                }
                6 => {
                    for i in 0..3 {
                        res[i] = ((parsed >> (16 - 8 * i) & 0xFF) as u8) as f64 / 255.;
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
            if c == 0. {
                0.
            } else if (v - rgb[0]).abs() < std::f64::EPSILON {
                std::f64::consts::FRAC_PI_3 * (0. + (rgb[1] - rgb[2]) / c)
            } else if (v - rgb[1]).abs() < std::f64::EPSILON {
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

            if l.abs() < std::f64::EPSILON || (l - 1.).abs() < std::f64::EPSILON {
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
                for l in 0..3 {
                    ciexyz_under_d65[l] += SRGB_TO_CIEXYZ[l][k] * gamma_expanded;
                }
            }
            for l in 0..3 {
                ciexyz_under_d65[l] = ciexyz_under_d65[l].clamp(0., 1.);
            }

            for k in 0..3 {
                for l in 0..3 {
                    ciexyz[[k, i, j]] += CHROMATIC_ADAPTATION_TRANSFORM[k][l] * ciexyz_under_d65[l];
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
            l.clamp(0., 100.) / 100.
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
                [0.5421119728870907, 80.98253915088577, 70.28651365985937],
                [0.8806247407807324, -79.21970052415361, 84.5923008745313],
                [0.28461577872330834, 71.28097404863831, -114.78144041108523],
                [1.0, 0.8195163861430821, 5.639091770210247],
                [0.0, 0.0, 0.0]
            ]]
        );
    }
}
