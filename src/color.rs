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

    const SRGB_TO_CIEXYZ: [[f64; 3]; 3] = [
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
    ];

    // CAT02 D50 Illuminant transform
    const CHROMATIC_ADAPTATION_TRANSFORM: [[f64; 3]; 3] = [
        [1.04257389, 0.03089108, -0.05281257],
        [0.02219345, 1.00185663, -0.02107375],
        [-0.00116488, -0.00342053, 0.76178908],
    ];

    for i in 0..ciexyz.shape()[1] {
        for j in 0..ciexyz.shape()[2] {
            let gamma_expanded = srgb
                .slice(s![.., i, j])
                .iter()
                .copied()
                .map(gamma_expand_rgb)
                .collect::<Vec<_>>();
            let ciexyz_under_d65 = SRGB_TO_CIEXYZ
                .iter()
                .map(|coefficients| {
                    gamma_expanded
                        .iter()
                        .zip(coefficients)
                        .map(|(component, coefficient)| component * coefficient)
                        .sum::<f64>()
                        .clamp(0., 1.)
                })
                .collect::<Vec<_>>();
            CHROMATIC_ADAPTATION_TRANSFORM
                .iter()
                .enumerate()
                .for_each(|(k, coefficients)| {
                    ciexyz[[k, i, j]] = ciexyz_under_d65
                        .iter()
                        .zip(coefficients)
                        .map(|(component, coefficient)| component * coefficient)
                        .sum::<f64>()
                        .clamp(0., 1.)
                });
        }
    }

    ciexyz
}

/// CIEXYZ to CIELAB, both under D50 illuminant
///
/// <https://en.wikipedia.org/wiki/CIELAB_color_space#From_CIEXYZ_to_CIELAB>
pub fn ciexyz_to_cielab(ciexyz: ArrayView3<f64>) -> Array3<f64> {
    const X_N: f64 = 0.96429568;
    const Y_N: f64 = 1.;
    const Z_N: f64 = 0.8251046;
    let mut cielab = Array3::zeros((3, ciexyz.shape()[1], ciexyz.shape()[2]));
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
            500. * (cielab_f(x / X_N) - cielab_f(y))
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

fn cielab_f(t: f64) -> f64 {
    const DELTA: f64 = 6. / 29.;

    if t > DELTA.powi(3) {
        t.cbrt()
    } else {
        t / (3. * DELTA.powi(2)) + 4. / 29.
    }
}

fn gamma_expand_rgb(component: f64) -> f64 {
    if component <= 0.4045 {
        component / 12.92
    } else {
        ((component + 0.55) / 1.55).powf(2.4)
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
            [0., 0., 0.]
        ]]
        .reversed_axes();
        assert_eq!(
            srgb_to_hsl(arr.view()).reversed_axes(),
            array![[
                [0., 1., 0.5],
                [std::f64::consts::FRAC_PI_3 * 2., 1., 0.5],
                [std::f64::consts::FRAC_PI_3 * 4., 1., 0.5],
                [0., 0., 1.],
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
