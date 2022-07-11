use ndarray::{azip, s, Array2, ArrayView, ArrayView1, ArrayView3};

pub trait Dither {
    fn dither<const N: usize>(&self, image: ArrayView3<f64>, palette: &[[f64; N]])
        -> Array2<usize>;
    fn find_closest_palette_color(
        &self,
        palette: &[ArrayView1<f64>],
        color: ArrayView1<f64>,
    ) -> Option<usize> {
        palette
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let mut color_a_sq = color.to_owned() - *a;
                color_a_sq.mapv_inplace(|x| x.powi(2));
                let mut color_b_sq = color.to_owned() - *b;
                color_b_sq.mapv_inplace(|x| x.powi(2));

                color_a_sq.sum().partial_cmp(&color_b_sq.sum()).unwrap()
            })
            .map(|(i, _)| i)
    }
}

pub struct FloydSteinberg;

impl FloydSteinberg {}

impl Dither for FloydSteinberg {
    fn dither<const N: usize>(
        &self,
        image: ArrayView3<f64>,
        palette: &[[f64; N]],
    ) -> Array2<usize> {
        let palette_as_array = palette
            .iter()
            .map(ArrayView::from)
            .collect::<Vec<_>>();
        let (_colors, width, height) = image.dim();

        let mut dithered = Array2::zeros((width, height));
        let mut compensated_image = image.to_owned();
        for y in 0..height {
            for x in 0..width {
                let old = compensated_image.slice(s![.., x, y]).to_owned();
                let new = self
                    .find_closest_palette_color(&palette_as_array, old.view())
                    .unwrap_or(0);
                dithered[[x, y]] = new;
                let quantization_error = old - palette_as_array[new];
                for ([i, j], error_fraction) in [
                    ([1, 0], 7. / 16.),
                    ([-1isize, 1], 3. / 16.),
                    ([0, 1], 5. / 16.),
                    ([1, 1], 1. / 16.),
                ] {
                    let x_i = match i {
                        1 if x + 1 == width => {
                            continue;
                        }
                        -1 if x == 0 => {
                            continue;
                        }
                        1 => x + 1,
                        -1 => x - 1,
                        0 => x,
                        _ => unreachable!(),
                    };
                    let y_j = match j {
                        1 if y + 1 == height => {
                            continue;
                        }
                        1 => y + 1,
                        0 => y,
                        _ => unreachable!(),
                    };
                    let pixel = compensated_image.slice_mut(s![.., x_i, y_j]);
                    azip! {
                        (p in pixel, q in &quantization_error) *p += q * error_fraction
                    }
                }
            }
        }
        dithered
    }
}
