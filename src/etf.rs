use lyon_geom::vector;
use lyon_geom::Angle;
use lyon_geom::Vector;
use ndarray::par_azip;
use ndarray::prelude::*;

/// Construct the edge tangent flow using the [Sobel operator](https://en.wikipedia.org/wiki/Sobel_operator)
///
/// <http://www.cs.umsl.edu/~kang/Papers/kang_tvcg09.pdf>
///
/// Sobel kernel is [Scharr's frequently kernel](https://en.wikipedia.org/wiki/Sobel_operator#Alternative_operators).
/// Edge handling is a kernel crop without compensation.
/// 
pub fn edge_tangent_flow(image: ArrayView2<f64>) -> Array2<Vector<f64>> {
    let sobel_x = array![[3., 0., -3.], [10., 0., -10.], [3., 0., -3.]];
    let mut t0 = Array::from_iter(
        conv_3x3(image.view(), sobel_x.view())
            .zip(conv_3x3(image.view(), sobel_x.t()))
            .map(|(x, y)| {
                // CCW perpendicular vectors
                let (sin, cos) = if x <= f64::EPSILON {
                    (1.0, 0.0)
                } else {
                    Angle::radians((y / x).atan() + std::f64::consts::FRAC_PI_2).sin_cos()
                };
                vector(x * cos - y * sin, x * sin + y * cos)
            }),
    )
    .into_shape((image.shape()[0], image.shape()[1]))
    .unwrap();

    // Magnitude
    let mut g_hat = t0.mapv(Vector::length);
    // Normalize magnitude
    let g_hat_max = g_hat
        .iter()
        .max_by(|x, y| x.partial_cmp(&y).unwrap())
        .copied()
        .unwrap();
    g_hat.par_mapv_inplace(|mag| mag / g_hat_max);

    // Normalize vectors
    t0.par_mapv_inplace(Vector::normalize);

    let mut t = t0;

    let slice_info_fn = |a, b| match (a, b) {
        (-1, -1) => (s![1.., 1..]),
        (0, -1) => s![.., 1..],
        (-1, 0) => s![1.., ..],
        (0, 0) => s![.., ..],
        (1, 0) => s![..-1, ..],
        (0, 1) => s![.., ..-1],
        (-1, 1) => s![1.., ..-1],
        (1, -1) => s![..-1, 1..],
        (1, 1) => s![..-1, ..-1],
        _ => unreachable!(),
    };

    for _ in 0..3 {
        let mut t_prime = Array::from_elem((image.shape()[0], image.shape()[1]), Vector::zero());
        for a in -1..=1isize {
            for b in -1..=1isize {
                let center_slice = slice_info_fn(a, b);
                let kernel_slice = slice_info_fn(-a, -b);
                par_azip! {(acc in &mut t_prime.slice_mut(center_slice), t_y in t.slice(kernel_slice), t_x in t.slice(center_slice), g_hat_y in g_hat.slice(kernel_slice), g_hat_x in g_hat.slice(center_slice)) {
                        let w_s = 1.; // By the nature of the kernel
                        let w_m = (1. + (g_hat_y - g_hat_x)) / 2.;
                        let dot_product = t_x.dot(*t_y);
                        let w_d = dot_product.abs();
                        let phi = dot_product.signum();
                        *acc += *t_y * phi * w_s * w_m * w_d;
                    }
                }
            }
        }
        t_prime.par_mapv_inplace(Vector::normalize);
        t = t_prime;
    }
    // Re-apply magnitude
    par_azip! {
        (v in &mut t, magnitude in &g_hat) {
            *v = *v * *magnitude;
        }
    }

    t
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
                let kernel_view = kernel.t();
                let mut kernel_it = kernel_view.iter();
                let mut acc = 0.;
                for a in i.saturating_sub(1)..=(i + 1).min(image.shape()[0] - 1) {
                    for b in j.saturating_sub(1)..=(j + 1).min(image.shape()[1] - 1) {
                        acc += kernel_it.next().unwrap() * image[[a, b]];
                    }
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
        let image = Array::from_elem((3, 3), 1.0);
        let kernel = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
        let result = Array::from_iter(conv_3x3(image.view(), kernel.view()))
            .into_shape((image.shape()[0], image.shape()[1]))
            .unwrap();
        assert_eq!(result[[2, 2]], (1..=9).sum::<usize>() as f64);
    }
}
