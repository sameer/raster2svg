use ndarray::prelude::*;
use ndarray::Array3;

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

/// sRGB under D65 illuminant to CIEXYZ under D50 illuminant
///
/// <https://en.wikipedia.org/wiki/SRGB#The_reverse_transformation_(sRGB_to_CIE_XYZ)>
///
/// <http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html>
pub fn srgb_to_ciexyz(srgb: ArrayView3<f64>) -> Array3<f64> {
    let mut ciexyz = Array3::zeros((3, srgb.shape()[1], srgb.shape()[2]));

    const X_COEFFICIENTS: [f64; 3] = [0.4360747, 0.3850649, 0.1430804];
    const Y_COEFFICIENTS: [f64; 3] = [0.2225045, 0.7168786, 0.0606169];
    const Z_COEFFICIENTS: [f64; 3] = [0.0139322, 0.0971045, 0.7141733];

    for i in 0..ciexyz.shape()[1] {
        for j in 0..ciexyz.shape()[2] {
            let gamma_expanded = srgb
                .slice(s![.., i, j])
                .iter()
                .copied()
                .map(gamma_expand_rgb)
                .collect::<Vec<_>>();
            [X_COEFFICIENTS, Y_COEFFICIENTS, Z_COEFFICIENTS]
                .iter()
                .enumerate()
                .for_each(|(k, coefficients)| {
                    ciexyz[[k, i, j]] = gamma_expanded
                        .iter()
                        .zip(coefficients)
                        .map(|(component, coefficient)| component * coefficient)
                        .sum::<f64>()
                        .clamp(0., 1.0);
                });
        }
    }

    ciexyz
}

/// CIEXYZ to CIELAB, both under D50 illuminant
///
/// <https://en.wikipedia.org/wiki/CIELAB_color_space#From_CIEXYZ_to_CIELAB>
pub fn ciexyz_to_cielab(ciexyz: ArrayView3<f64>) -> Array3<f64> {
    let mut cielab = Array3::zeros((3, ciexyz.shape()[1], ciexyz.shape()[2]));
    cielab
        .slice_mut(s![0, .., ..])
        .assign(&ciexyz.map_axis(Axis(0), |xyz| {
            let y = xyz[1];
            let l = 116. * cielab_f(y) - 16.;
            l.clamp(0., 100.) / 100.
        }));

    cielab
        .slice_mut(s![1, .., ..])
        .assign(&ciexyz.map_axis(Axis(0), |xyz| {
            let x = xyz[0];
            let y = xyz[1];
            500. * (cielab_f(x / 0.964212) - cielab_f(y))
        }));

    cielab
        .slice_mut(s![2, .., ..])
        .assign(&ciexyz.map_axis(Axis(0), |xyz| {
            let y = xyz[1];
            let z = xyz[2];
            200. * (cielab_f(y) - cielab_f(z / 0.825188))
        }));

    cielab
}

fn cielab_f(t: f64) -> f64 {
    const DELTA: f64 = 6. / 29.;

    if t > DELTA.powi(3) {
        t.cbrt()
    } else {
        t / (3. * DELTA.powi(2)) + 4. / 29.
    }
}

fn gamma_expand_rgb(component: f64) -> f64 {
    if component <= 0.04045 {
        component / 12.92
    } else {
        ((component + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
#[test]
fn check_srgb_to_cielab() {
    let arr = array![[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0]
    ]]
    .reversed_axes();
    dbg!(ciexyz_to_cielab(srgb_to_ciexyz(arr.view()).view()).reversed_axes());
    dbg!(srgb_to_hsl(arr.view()).reversed_axes());
}
