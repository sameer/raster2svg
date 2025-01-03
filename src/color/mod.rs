use ndarray::prelude::*;
use ndarray::Array3;
use std::fmt;
use std::fmt::Display;
use std::num::ParseIntError;
use std::ops::Index;
use std::str::FromStr;

use self::transform::{cielab_to_ciehcl, ciexyz_to_cielab, srgb_to_ciexyz, srgb_to_hsl};
use crate::ColorModel;

/// Color space transformations.
mod transform;

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

    /// CIEDE2000 Perceptual color distance metric.
    ///
    /// CIEHCL: based on CIEDE2000
    ///
    /// <https://en.wikipedia.org/wiki/Color_difference#CIEDE2000>
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

/// Convert a 1D Rust array into its (N, 1, 1) [ndarray] equivalent.
fn a_to_nd<const N: usize>(x: &[f64; N]) -> Array3<f64> {
    Array3::<f64>::from_shape_vec((N, 1, 1), x.as_ref().to_vec()).unwrap()
}

/// Convert a (N, 1, 1) [ndarray] array into its Rust equivalent.
fn nd_to_a<const N: usize>(a: Array3<f64>) -> [f64; N] {
    let view = a.slice(s![.., 0, 0]);
    let mut res = [0.; N];
    for i in 0..N {
        res[i] = view[i];
    }
    res
}
