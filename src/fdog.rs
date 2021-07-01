use lyon_geom::vector;
use lyon_geom::Vector;
use ndarray::par_azip;
use ndarray::prelude::*;

use crate::get_slice_info_for_offset;

/// Construct the edge tangent flow using the [Sobel operator](https://en.wikipedia.org/wiki/Sobel_operator)
///
/// Sobel operator kernel is [Scharr's frequently used kernel](https://en.wikipedia.org/wiki/Sobel_operator#Alternative_operators).
///
/// Edge handling is a kernel crop without compensation.
///
/// <http://umsl.edu/cmpsci/about/People/Faculty/HenryKang/coon.pdf>
/// <http://www.cs.umsl.edu/~kang/Papers/kang_tvcg09.pdf>
///
pub fn edge_tangent_flow(image: ArrayView2<f64>) -> Array2<Vector<f64>> {
    let [width, height] = if let [width, height] = *image.shape() {
        [width, height]
    } else {
        unreachable!()
    };

    let sobel_x = array![[3., 0., -3.], [10., 0., -10.], [3., 0., -3.]];
    let sobel_y = sobel_x.t();

    // Magnitude and initial ETF based on Sobel operator
    let (mut ĝ, mut t) = {
        let t0: Array2<Vector<f64>> = Array::from_iter(
            conv_3x3(image.view(), sobel_x.view())
                .zip(conv_3x3(image.view(), sobel_y))
                .map(|(x, y)| {
                    // CCW perpendicular vectors
                    vector(-y, x)
                }),
        )
        .into_shape((width, height))
        .unwrap();
        (t0.mapv(Vector::length), t0)
    };
    // Normalization
    {
        let ĝ_max = ĝ
            .iter()
            .max_by(|x, y| x.partial_cmp(&y).unwrap())
            .copied()
            .unwrap();
        ĝ.par_mapv_inplace(|ĝ_x| ĝ_x / ĝ_max);
        t.par_mapv_inplace(Vector::normalize);
    }

    let radius = 3;

    for _ in 0..3 {
        let mut t_prime = Array2::from_elem((width, height), Vector::zero());
        for i in -radius..=radius {
            for j in -radius..=radius {
                let center_slice = get_slice_info_for_offset(i, j);
                let kernel_slice = get_slice_info_for_offset(-i, -j);
                par_azip! {(t_prime_x in &mut t_prime.slice_mut(center_slice), t_y in t.slice(kernel_slice), t_x in t.slice(center_slice), ĝ_y in ĝ.slice(kernel_slice), ĝ_x in ĝ.slice(center_slice)) {
                        let w_s = if i.pow(2) + j.pow(2) < radius.pow(2) { 1. } else { 0. };
                        // Some implementations use tanh here. This is only required if the fall-off rate is greater than 1.
                        let w_m = (ĝ_y - ĝ_x + 1.) / 2.;
                        // Note that due to normalization, this is actually just the cosine of the angle between the vectors.
                        let dot_product = t_x.dot(*t_y);
                        let w_d = dot_product.abs();
                        let phi = dot_product.signum();
                        *t_prime_x += *t_y * phi * w_s * w_m * w_d;
                    }
                }
            }
        }
        t_prime.par_mapv_inplace(Vector::normalize);
        t.assign(&t_prime);
    }

    t
}

/// Apply the FDoG filter to the image using the edge tangent flow.
///
/// Uses parameter values recommended in the paper.
///
/// <http://umsl.edu/cmpsci/about/People/Faculty/HenryKang/coon.pdf>
/// <http://www.cs.umsl.edu/~kang/Papers/kang_tvcg09.pdf>
pub fn flow_based_difference_of_gaussians(
    image: ArrayView2<f64>,
    etf: ArrayView2<Vector<f64>>,
) -> Array2<f64> {
    const NUM_STANDARD_DEVIATIONS: f64 = 3.;

    let sigma_c = 1.0;
    let sigma_s = 1.6 * sigma_c;
    let t_range = (NUM_STANDARD_DEVIATIONS * sigma_s).ceil() as usize;
    let rho = 0.99;
    let sigma_m = 3.0;
    let s_range = (NUM_STANDARD_DEVIATIONS * sigma_m).ceil() as usize;
    let tau = 0.5;

    let [width, height] = if let [width, height] = *image.shape() {
        [width, height]
    } else {
        unreachable!()
    };

    let positions = Array::from_iter(
        (0..width)
            .map(|x| (0..height).map(move |y| [x, y]))
            .flatten(),
    )
    .into_shape((width, height))
    .unwrap();

    let mut i = image.to_owned();
    let mut ĥ = Array2::zeros((width, height));

    for _ in 0..3 {
        let mut integrated_over_t = Array2::<f64>::zeros((width, height));
        par_azip! {
            (i_t in &mut integrated_over_t, position in &positions) {
                let mut f_t_sum = 0.;
                for direction in [-1.0, 1.0] {
                    let mut position = *position;
                    // Rotate ETF vector 90° clockwise
                    let gradient_tangent_vector = vector(etf[position].y, -etf[position].x) * direction;

                    for t in 0..=t_range {

                        if !(t == 0 && direction > 0.0) {
                            let f_t = gaussian_pdf(t as f64, sigma_c) - rho * gaussian_pdf(t as f64, sigma_s);
                            f_t_sum += f_t;
                            *i_t += i[position] * f_t;
                        }

                        let mut reached_edge_of_image = false;

                        for ((dim_dir, dim_position), dim_limit) in gradient_tangent_vector.to_array().iter().zip(position.iter_mut()).zip([width, height]) {
                            let dim_dir = dim_dir.round();
                            if dim_dir < 0. {
                                if *dim_position == 0 {
                                    reached_edge_of_image = true;
                                } else {
                                    *dim_position -= 1;
                                }
                            }
                            else if dim_dir > 0. {
                                if *dim_position + 1 == dim_limit {
                                    reached_edge_of_image = true;
                                } else {
                                    *dim_position += 1;
                                }
                            }
                        }
                        if reached_edge_of_image {
                            break
                        }
                    }
                }
                *i_t /= f_t_sum as f64;
            }
        };

        let mut h = Array2::<f64>::zeros((width, height));

        par_azip! {
            (h_s in &mut h, position in &positions) {
                let mut g_m_sum = 0.;
                for direction in [-1.0, 1.0] {
                    let mut position = *position;
                    for s in 0..=s_range {
                        // An important distinction here: as opposed to the gradient_tangent vector,
                        // the flow vector is NOT fixed and changes as the position is updated.
                        let flow_vector = etf[position] * direction;

                        if !(s == 0 && direction > 0.0) {
                            let g_m = gaussian_pdf(s as f64, sigma_m);
                            g_m_sum += g_m;
                            *h_s += integrated_over_t[position] * g_m;
                        }

                        let mut reached_edge_of_image = false;

                        for ((dim_dir, dim_position), dim_limit) in flow_vector.to_array().iter().zip(position.iter_mut()).zip([width, height]) {
                            let dim_dir = dim_dir.round();
                            if dim_dir < 0. {
                                if *dim_position == 0 {
                                    reached_edge_of_image = true;
                                } else {
                                    *dim_position -= 1;
                                }
                            }
                            else if dim_dir > 0.{
                                if *dim_position + 1 == dim_limit {
                                    reached_edge_of_image = true;
                                } else {
                                    *dim_position += 1;
                                }
                            }
                        }

                        if reached_edge_of_image {
                            break
                        }
                    }
                }
                *h_s /= g_m_sum as f64;
            }
        };

        par_azip! {
            (ĥ_x in &mut ĥ, h_x in &h) {
                *ĥ_x = if *h_x < 0. && 1. + h_x.tanh() < tau {
                    0.
                } else {
                    1.
                };
            }
        };

        par_azip! {
            (i_x in &mut i, ĥ_x in &ĥ) {
                *i_x = i_x.min(*ĥ_x);
            }
        };
    }

    ĥ
}

/// <https://en.wikipedia.org/wiki/Normal_distribution>
#[inline]
fn gaussian_pdf(x: f64, sigma: f64) -> f64 {
    (-0.5 * (x / sigma).powi(2)).exp()
        / (std::f64::consts::PI.sqrt() * std::f64::consts::SQRT_2 * sigma)
}

/// Use a 3x3 [kernel](https://en.wikipedia.org/wiki/Kernel_(image_processing)) to do convolution on an image.
///
/// Edge handling is a kernel crop without compensation.
fn conv_3x3<'a>(
    image: ArrayView2<'a, f64>,
    kernel: ArrayView2<'a, f64>,
) -> impl Iterator<Item = f64> + 'a {
    (0..image.shape()[0])
        .map(move |i| {
            (0..image.shape()[1]).map(move |j| {
                let kernel_transpose = kernel.t();
                let mut kernel_it = kernel_transpose.iter();
                let mut acc = 0.;
                let x_range = i.saturating_sub(1)..=(i + 1).min(image.shape()[0] - 1);
                if i == 0 {
                    kernel_it.next();
                }
                for x in x_range {
                    let y_range = j.saturating_sub(1)..=(j + 1).min(image.shape()[1] - 1);
                    if j == 0 {
                        kernel_it.next();
                    }
                    for y in y_range {
                        acc += kernel_it.next().unwrap() * image[[x, y]];
                    }
                    if j == image.shape()[0] - 1 {
                        kernel_it.next();
                    }
                }
                if i == image.shape()[0] - 1 {
                    kernel_it.next();
                }
                acc
            })
        })
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_convolution_operator() {
        let image = Array2::ones((3, 3));
        let kernel = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
        let result = Array::from_iter(conv_3x3(image.view(), kernel.view()))
            .into_shape((image.shape()[0], image.shape()[1]))
            .unwrap();
        assert_eq!(result[[1, 1]], (1..=9).sum::<usize>() as f64);
    }
}
