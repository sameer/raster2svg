use lyon_geom::point;
use lyon_geom::vector;
use lyon_geom::Line;
use lyon_geom::Vector;
use ndarray::par_azip;
use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_stats::QuantileExt;

use crate::get_slice_info_for_offset;

/// 99.73% of values are accounted for within 3 standard deviations
///
/// <https://en.wikipedia.org/wiki/Normal_distribution#/media/File:Standard_deviation_diagram.svg>
const NUM_STANDARD_DEVIATIONS: f64 = 3.;

/// [Sobel operator](https://en.wikipedia.org/wiki/Sobel_operator)
///
/// Edge handling is a kernel crop without compensation.
pub fn sobel_operator(image: ArrayView2<f64>) -> Array2<Vector<f64>> {
    /// [Scharr operator](https://en.wikipedia.org/wiki/Sobel_operator#Alternative_operators) with better rotational symmetry.
    const SOBEL_X: [f64; 9] = [3., 0., -3., 10., 0., -10., 3., 0., -3.];

    let sobel_x = ArrayView2::from_shape((3, 3), &SOBEL_X).unwrap();
    let sobel_y = sobel_x.t();
    let g_x = convolve(image.view(), sobel_x.view());
    let g_y = convolve(image.view(), sobel_y);

    Zip::from(&g_x)
        .and(&g_y)
        .par_map_collect(|g_x, g_y| vector(*g_x, *g_y))
}

/// Construct the edge tangent flow.
///
/// Array of the vectors tangent to the gradient at each pixel.
///
/// Edge handling is a kernel crop without compensation.
///
/// <http://umsl.edu/cmpsci/about/People/Faculty/HenryKang/coon.pdf>
/// <http://www.cs.umsl.edu/~kang/Papers/kang_tvcg09.pdf>
///
pub fn edge_tangent_flow(image: ArrayView2<f64>) -> Array2<Vector<f64>> {
    let radius = 3;
    let iterations = 3;

    // Magnitude and initial ETF based on Sobel operator
    let (mut ĝ, mut t) = {
        let mut t0 = sobel_operator(image);
        // CCW vector is tangent to the gradient
        t0.mapv_inplace(|v| vector(-v.y, v.x));
        (t0.mapv(Vector::length), t0)
    };
    // Normalization
    {
        let ĝ_max = *ĝ.max().unwrap();
        ĝ.par_mapv_inplace(|ĝ_x| ĝ_x / ĝ_max);
        t.par_mapv_inplace(Vector::normalize);
    }

    for _ in 0..iterations {
        let mut t_prime = Array2::from_elem(image.raw_dim(), Vector::zero());
        for i in -radius..=radius {
            for j in -radius..=radius {
                let center_slice = get_slice_info_for_offset(-i, -j);
                let kernel_slice = get_slice_info_for_offset(i, j);
                // w_s determines whether this computation is useful
                let w_s = i.pow(2) + j.pow(2) < radius.pow(2);
                if w_s {
                    par_azip! {(t_prime_x in &mut t_prime.slice_mut(center_slice), t_y in t.slice(kernel_slice), t_x in t.slice(center_slice), ĝ_y in ĝ.slice(kernel_slice), ĝ_x in ĝ.slice(center_slice)) {
                            // Some implementations use tanh here. This is only required if the fall-off rate is greater than 1.
                            let w_m = (ĝ_y - ĝ_x + 1.) / 2.;
                            // Note that due to normalization, this is actually just the cosine of the angle between the vectors.
                            let dot_product = t_x.dot(*t_y);
                            let w_d = dot_product.abs();
                            let phi = dot_product.signum();
                            *t_prime_x += *t_y * phi * w_m * w_d;
                        }
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
    let sigma_c = 1.;
    let sigma_s = 1.6 * sigma_c;
    let t_range = (NUM_STANDARD_DEVIATIONS * sigma_s).ceil() as usize;
    let rho = 0.99;
    let sigma_m = 3.;
    let s_range = (NUM_STANDARD_DEVIATIONS * sigma_m).ceil() as usize;
    let tau = 0.5;
    let iterations = 3;

    let (width, height) = image.dim();

    let positions = Array::from_iter((0..width).flat_map(|x| (0..height).map(move |y| [x, y])))
        .into_shape_clone((width, height))
        .unwrap();

    let mut i = image.to_owned();
    let mut ĥ = Array2::zeros((width, height));

    for _ in 0..iterations {
        let mut integrated_over_t = Array2::<f64>::zeros((width, height));
        Zip::from(&mut integrated_over_t)
            .and(&positions)
            .par_for_each(|i_x, position| {
                let mut f_t_sum = 0.;
                // Rotate ETF vector 90° clockwise
                let gradient_perpendicular_vector =
                    vector(etf[*position].y, -etf[*position].x).normalize();
                let iterate_by_y =
                    gradient_perpendicular_vector.y.abs() > gradient_perpendicular_vector.x.abs();
                let line_equation = Line {
                    point: point(position[0] as f64, position[1] as f64),
                    vector: gradient_perpendicular_vector,
                }
                .equation();
                for direction in [-1.0, 1.0] {
                    let mut integration_position = *position;

                    for t in 0..=t_range {
                        if !(t == 0 && direction > 0.0) {
                            let f_t = gaussian_pdf(t as f64, sigma_c)
                                - rho * gaussian_pdf(t as f64, sigma_s);
                            f_t_sum += f_t;
                            *i_x += i[integration_position] * f_t;
                        }

                        let mut reached_edge_of_image = false;

                        // Bresenham's line algorithm
                        let solved = if iterate_by_y {
                            let y = position[1] as f64 + (t + 1) as f64 * direction;
                            [line_equation.solve_x_for_y(y).unwrap(), y]
                        } else {
                            let x = position[0] as f64 + (t + 1) as f64 * direction;
                            [x, line_equation.solve_y_for_x(x).unwrap()]
                        };

                        for ((dim_position, dim_solved), dim_limit) in integration_position
                            .iter_mut()
                            .zip(solved.iter())
                            .zip([width, height])
                        {
                            let dim_solved = dim_solved.round();
                            if dim_solved < 0. || dim_solved >= (dim_limit - 1) as f64 {
                                reached_edge_of_image = true;
                                break;
                            } else {
                                *dim_position = dim_solved as usize;
                            }
                        }
                        if reached_edge_of_image {
                            break;
                        }
                    }
                }
                *i_x /= f_t_sum;
            });

        let mut h = Array2::<f64>::zeros((width, height));

        Zip::from(&mut h)
            .and(&positions)
            .par_for_each(|h_x, position| {
                let mut g_m_sum = 0.;
                for direction in [-1.0, 1.0] {
                    let mut integration_position = *position;
                    for s in 0..=s_range {
                        // An important distinction here: as opposed to the gradient_perpendicular_vector,
                        // the flow_vector is NOT fixed and changes as the position is updated to follow the curve.
                        let flow_vector = etf[integration_position];

                        let iterate_by_y = flow_vector.y.abs() > flow_vector.x.abs();
                        let line_equation = Line {
                            point: point(
                                integration_position[0] as f64,
                                integration_position[1] as f64,
                            ),
                            vector: flow_vector,
                        }
                        .equation();

                        if !(s == 0 && direction > 0.0) {
                            let g_m = gaussian_pdf(s as f64, sigma_m);
                            g_m_sum += g_m;
                            *h_x += integrated_over_t[integration_position] * g_m;
                        }

                        let mut reached_edge_of_image = false;

                        // Bresenham's line algorithm
                        let solved = if iterate_by_y {
                            let y = integration_position[1] as f64 + direction;
                            [line_equation.solve_x_for_y(y).unwrap(), y]
                        } else {
                            let x = integration_position[0] as f64 + direction;
                            [x, line_equation.solve_y_for_x(x).unwrap()]
                        };

                        for ((dim_position, dim_solved), dim_limit) in integration_position
                            .iter_mut()
                            .zip(solved.iter())
                            .zip([width, height])
                        {
                            let dim_solved = dim_solved.round();
                            if dim_solved < 0. || dim_solved >= (dim_limit - 1) as f64 {
                                reached_edge_of_image = true;
                                break;
                            } else {
                                *dim_position = dim_solved as usize;
                            }
                        }
                        if reached_edge_of_image {
                            break;
                        }
                    }
                }
                *h_x /= g_m_sum;
            });

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

pub struct EdgeFlowEstimate {
    /// measure of local contrast
    pub eigenvalues: Array2<[f64; 2]>,
    /// direction of maximum and minimum local contrast
    pub eigenvectors: Array2<[Vector<f64>; 2]>,
    /// ranges from 0 (isotropic) to 1 (strongly oriented)
    pub local_anisotropy: Array2<f64>,
}

/// Edge flow estimation using local contrast eigenvectors
///
/// This expects an input image to be smoothed with a Gaussian filter.
/// Otherwise the eigenvectors will have a high degree of discontinuity.
///
/// <https://zero.sci-hub.st/1315/448d22408920f4a25dbe42317a9dde02/wang2012.pdf#bm_2212_st4>
pub fn edge_flow_estimation(image: ArrayView2<f64>) -> EdgeFlowEstimate {
    let gradient_vectors: Array2<Vector<f64>> = sobel_operator(image);
    let eigenvalues = gradient_vectors.mapv(|v| {
        let e = v.x.powi(2);
        let f = v.x * v.y;
        let g = v.y.powi(2);
        let e_g_sum = e + g;
        let sqrt_sum = ((e - g).powi(2) + 4. * f.powi(2)).sqrt();
        [(e_g_sum + sqrt_sum) / 2., (e_g_sum - sqrt_sum) / 2.]
    });
    let eigenvectors = Zip::from(&gradient_vectors)
        .and(&eigenvalues)
        .map_collect(|v, e| {
            [
                vector(v.x * v.y, e[0] - v.x.powi(2)),
                vector(e[1] - v.y.powi(2), v.x * v.y),
            ]
        });
    let local_anisotropy = eigenvalues.mapv(|e| (e[0] - e[1]) / (e[0] + e[1]));
    EdgeFlowEstimate {
        eigenvalues,
        eigenvectors,
        local_anisotropy,
    }
}

/// Step edge detection using the edge flow estimate for conditioning.
///
/// <https://zero.sci-hub.st/1315/448d22408920f4a25dbe42317a9dde02/wang2012.pdf#bm_2212_st3>
pub fn step_edge_detection(
    image: ArrayView2<f64>,
    edge_flow_estimate: ArrayView2<[Vector<f64>; 2]>,
) -> Array2<f64> {
    let sigma_c = 1.;
    let sigma_s = 1.6 * sigma_c;
    let rho = 1.;
    let phi_e_sharpness = 0.25;
    let threshold = 0.3;

    let (width, height) = image.dim();

    let t_range = (NUM_STANDARD_DEVIATIONS * sigma_s).ceil() as usize;

    let positions = Array::from_iter((0..width).flat_map(|x| (0..height).map(move |y| [x, y])))
        .into_shape_clone(image.raw_dim())
        .unwrap();

    let mut d = Array2::<f64>::zeros(image.raw_dim());
    Zip::from(&mut d)
        .and(&positions)
        .par_for_each(|d_x, position| {
            let mut first_derivative_component = 0.;
            let mut first_derivative_weight_sum = 0.;
            let mut laplacian_of_gaussian_component = 0.;
            let mut laplacian_of_gaussian_weight_sum = 0.;

            let gradient_perpendicular_vector = edge_flow_estimate[*position][0].normalize();
            let iterate_by_y =
                gradient_perpendicular_vector.y.abs() > gradient_perpendicular_vector.x.abs();
            let line_equation = Line {
                point: point(position[0] as f64, position[1] as f64),
                vector: gradient_perpendicular_vector,
            }
            .equation();
            for direction in [-1.0, 1.0] {
                let mut sum_position = *position;

                for t in 0..=t_range {
                    if !(t == 0 && direction > 0.0) {
                        let dist = (vector(position[0] as f64, position[1] as f64)
                            - vector(sum_position[0] as f64, sum_position[1] as f64))
                        .length();
                        let first_derivative_weight = gaussian_pdf_first_derivative(dist, sigma_c);
                        first_derivative_weight_sum += first_derivative_weight;
                        first_derivative_component += first_derivative_weight * image[sum_position];

                        let laplacian_of_gaussian_weight =
                            gaussian_pdf(dist, sigma_c) - rho * gaussian_pdf(dist, sigma_s);
                        laplacian_of_gaussian_weight_sum += laplacian_of_gaussian_weight;
                        laplacian_of_gaussian_component +=
                            laplacian_of_gaussian_weight * image[sum_position];
                    }

                    let mut reached_edge_of_image = false;

                    // Bresenham's line algorithm
                    let solved = if iterate_by_y {
                        let y = position[1] as f64 + (t + 1) as f64 * direction;
                        [line_equation.solve_x_for_y(y).unwrap(), y]
                    } else {
                        let x = position[0] as f64 + (t + 1) as f64 * direction;
                        [x, line_equation.solve_y_for_x(x).unwrap()]
                    };

                    for ((dim_position, dim_solved), dim_limit) in sum_position
                        .iter_mut()
                        .zip(solved.iter())
                        .zip([width, height])
                    {
                        let dim_solved = dim_solved.round();
                        if dim_solved < 0. || dim_solved >= (dim_limit - 1) as f64 {
                            reached_edge_of_image = true;
                            break;
                        } else {
                            *dim_position = dim_solved as usize;
                        }
                    }
                    if reached_edge_of_image {
                        break;
                    }
                }
            }
            *d_x = first_derivative_component.abs() - laplacian_of_gaussian_component.abs();
        });

    #[allow(clippy::let_and_return)]
    let h = {
        // let min = *d.min().unwrap();
        // let max = *d.max().unwrap();
        d.mapv_inplace(|d_x| {
            // (d_x - min) / (max - min)
            if d_x < threshold {
                1.
            } else {
                1. - (phi_e_sharpness * d_x).tanh()
            }
        });
        d
    };

    h
}

/// <https://en.wikipedia.org/wiki/Normal_distribution>
#[inline]
fn gaussian_pdf(x: f64, sigma: f64) -> f64 {
    (-0.5 * (x / sigma).powi(2)).exp()
        / (std::f64::consts::PI.sqrt() * std::f64::consts::SQRT_2 * sigma)
}

#[inline]
fn gaussian_pdf_first_derivative(x: f64, sigma: f64) -> f64 {
    (-0.5 * (x / sigma).powi(2)).exp() * -x
        / (std::f64::consts::PI.sqrt() * std::f64::consts::SQRT_2 * sigma.powi(3))
}

#[inline]
fn gaussian_cdf(x: f64, sigma: f64) -> f64 {
    0.5 * (1. + erf(x / (sigma * std::f64::consts::SQRT_2)))
}

fn erf(x: f64) -> f64 {
    let sign = x.signum();
    let x = x.abs();
    let p = 0.47047;
    let a1 = 0.3480242;
    let a2 = -0.0958798;
    let a3 = 0.7478556;
    let t = 1. / (1. + p * x);
    let tau = (a1 * t + a2 * t.powi(2) + a3 * t.powi(3)) * (-x.powi(2)).exp();
    if sign >= 0. {
        1. - tau
    } else {
        tau - 1.
    }
}

/// Use an NxN [kernel](https://en.wikipedia.org/wiki/Kernel_(image_processing)) to do convolution on an image.
///
/// Edge handling is a kernel crop without compensation.
fn convolve<'a>(image: ArrayView2<'a, f64>, kernel: ArrayView2<'a, f64>) -> Array2<f64> {
    let kernel_transpose = kernel.t();
    let (kernel_width, kernel_height) = kernel.raw_dim().into_pattern();
    let mut it = kernel_transpose.iter();
    let mut convolved = Array::zeros(image.raw_dim());
    for i in -(kernel_width as i32) / 2..=kernel_width as i32 / 2 {
        for j in -(kernel_height as i32) / 2..=kernel_height as i32 / 2 {
            let center_slice = get_slice_info_for_offset(-i, -j);
            let kernel_slice = get_slice_info_for_offset(i, j);
            let coefficient = it.next().unwrap();
            Zip::from(convolved.slice_mut(center_slice))
                .and(image.slice(kernel_slice))
                .par_for_each(|dest, kernel| *dest += coefficient * *kernel);
        }
    }
    convolved
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_convolution_operator() {
        let image = Array2::ones((3, 3));
        let kernel = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
        let result = convolve(image.view(), kernel.view());
        assert_eq!(result[[1, 1]], (1..=9).sum::<usize>() as f64);
    }
}
