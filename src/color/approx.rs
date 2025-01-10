use lyon_geom::euclid::default::Vector3D;
use ndarray::{s, Array, Array3, ArrayView1, ArrayView3};
use tracing::debug;

use crate::{kbn_summation, optimize::adc_direct::AdcDirect, ColorModel};

use super::Color;

impl ColorModel {
    pub fn approximate(self, image: ArrayView3<f64>, palette: &[Color]) -> Array3<f64> {
        let (_, width, height) = image.dim();

        let mut image_in_cylindrical_color_model = {
            let image_in_color_model = self.convert(image);
            self.cylindrical(image_in_color_model.view())
        };
        let implements_in_color_model = palette
            .iter()
            .map(|c| self.convert_single(c))
            .collect::<Vec<_>>();
        let mut implements_in_cylindrical_color_model = implements_in_color_model
            .into_iter()
            .map(|c| self.cylindrical_single(c))
            .collect::<Vec<_>>();

        let white_in_cylindrical_color_model =
            self.cylindrical_single(self.convert_single(&Color::from([1.; 3])));

        // Convert lightness into darkness and clamp it
        image_in_cylindrical_color_model
            .slice_mut(s![2, .., ..])
            .mapv_inplace(|lightness| (white_in_cylindrical_color_model[2] - lightness).max(0.));
        implements_in_cylindrical_color_model
            .iter_mut()
            .for_each(|[_, _, lightness]| {
                *lightness = (white_in_cylindrical_color_model[2] - *lightness).max(0.);
            });

        let implement_hue_vectors = implements_in_cylindrical_color_model
            .into_iter()
            .map(|[hue, magnitude, darkness]| {
                let (sin, cos) = hue.sin_cos();
                Vector3D::new(cos * magnitude, sin * magnitude, darkness)
            })
            .collect::<Vec<_>>();

        let mut image_in_implements = Array3::<f64>::zeros((palette.len(), width, height));
        // let mut cached_colors = FxHashMap::default();
        for y in 0..height {
            debug!(y);
            for x in 0..width {
                let desired: [f64; 3] = image_in_cylindrical_color_model
                    .slice(s![.., x, y])
                    .to_vec()
                    .try_into()
                    .expect("image slice is a pixel");

                let direct = AdcDirect {
                    function: self.objective_function(desired, &implement_hue_vectors),
                    bounds: Array::from_elem(implement_hue_vectors.len(), [0., 1.]),
                    // max_evaluations: None,
                    // max_iterations: Some(100),
                    max_evaluations: Some(10_000),
                    max_iterations: None,
                };

                let (best, _best_cost) = direct.run();

                image_in_implements
                    .slice_mut(s![.., x, y])
                    .assign(&best.view());
            }
        }

        image_in_implements
    }

    pub fn objective_function(
        self,
        desired: [f64; 3],
        implement_hue_vectors: &'_ [Vector3D<f64>],
    ) -> impl Fn(ArrayView1<f64>) -> f64 + '_ {
        move |param| {
            kbn_summation! {
                for i in 0..implement_hue_vectors.len() => {
                    weighted_vector_x += implement_hue_vectors[i].x * param[i];
                    weighted_vector_y += implement_hue_vectors[i].y * param[i];
                    weighted_vector_z += implement_hue_vectors[i].z * param[i];
                }
            }
            let weighted_vector =
                Vector3D::<f64>::from([weighted_vector_x, weighted_vector_y, weighted_vector_z]);
            // Convert back to cylindrical model (hue, chroma, darkness)
            let actual = [
                weighted_vector.y.atan2(weighted_vector.x),
                weighted_vector.to_2d().length(),
                weighted_vector.z,
            ];
            ColorModel::Cielab.cylindrical_diff(desired, actual)
        }
    }
}
