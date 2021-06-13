use ndarray::prelude::*;
use ndarray::Array3;

pub fn srgb_to_hsl(srgb: &Array3<f64>) -> Array3<f64> {
    let mut hsl = Array::zeros((3, srgb.shape()[1], srgb.shape()[2]));
    hsl.slice_mut(s![0, .., ..])
        .assign(&srgb.map_axis(Axis(0), |rgb| {
            let v = rgb[0].max(rgb[1]).max(rgb[2]);
            let c = v - rgb[0].min(rgb[1]).min(rgb[2]);
            if c == 0. {
                0.
            } else if v == rgb[0] {
                std::f64::consts::FRAC_PI_3 * (0. + (rgb[1] - rgb[2]) / c)
            } else if v == rgb[1] {
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

            if l == 0. || l == 1. {
                0.
            } else {
                (v - l) / (l.min(1. - l))
            }
        }));

    hsl.slice_mut(s![2, .., ..])
        .assign(&srgb.map_axis(Axis(0), |rgb| {
            let v = rgb[0].max(rgb[1]).max(rgb[2]);
            let c = v - rgb[0].min(rgb[1]).min(rgb[2]);
            let l = v - c / 2.;
            l
        }));

    hsl
}

/// Convert sRGB under D65 illuminant to CIEXYZ under D50 illuminant
pub fn srgb_to_ciexyz(srgb: &Array3<f64>) -> Array3<f64> {
    let mut ciexyz = Array::zeros((3, srgb.shape()[1], srgb.shape()[2]));

    const X_COEFFICIENTS: [f64; 3] = [0.4360747, 0.3850649, 0.1430804];
    const Y_COEFFICIENTS: [f64; 3] = [0.2225045, 0.7168786, 0.0606169];
    const Z_COEFFICIENTS: [f64; 3] = [0.0139322, 0.0971045, 0.7141733];
    [X_COEFFICIENTS, Y_COEFFICIENTS, Z_COEFFICIENTS]
        .iter()
        .enumerate()
        .for_each(|(i, coefficients)| {
            ciexyz
                .slice_mut(s![i, .., ..])
                .assign(&srgb.map_axis(Axis(0), |rgb| {
                    rgb.iter()
                        .copied()
                        .map(gamma_expand_rgb)
                        .zip(coefficients)
                        .map(|(component, coefficient)| component * coefficient)
                        .sum::<f64>()
                        .clamp(0., 1.0)
                }));
        });

    ciexyz
}

/// CIEXYZ to CIELAB both under D50 illuminant
pub fn ciexyz_to_cielab(ciexyz: &Array3<f64>) -> Array3<f64> {
    let mut cielab = Array::zeros((3, ciexyz.shape()[1], ciexyz.shape()[2]));
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
            let a = 500. * (cielab_f(x / 0.964212) - cielab_f(y));
            // a.clamp(0., 100.) / 100.
            a
        }));

    cielab
        .slice_mut(s![2, .., ..])
        .assign(&ciexyz.map_axis(Axis(0), |xyz| {
            let y = xyz[1];
            let z = xyz[2];
            let b = 200. * (cielab_f(y) - cielab_f(z / 0.825188));
            b
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
    dbg!(ciexyz_to_cielab(&srgb_to_ciexyz(&arr)).reversed_axes());
    dbg!(srgb_to_hsl(&arr).reversed_axes());
}
