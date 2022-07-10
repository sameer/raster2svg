use ndarray::{azip, s, stack, Array1, Array2, ArrayView1, Axis};

use crate::kbn_summation;

pub struct Direct<F>
where
    F: Fn(ArrayView1<f64>) -> f64,
{
    pub epsilon: f64,
    pub max_evaluations: Option<usize>,
    pub max_iterations: Option<usize>,
    pub initial: Array1<f64>,
    pub bounds: Array1<[f64; 2]>,
    pub function: F,
}

#[derive(Debug)]
struct Rectangle {
    bound_ranges: Array1<f64>,
    size: f64,
    center: Array1<f64>,
    fmin: f64,
}

#[derive(Debug)]
struct Group {
    size: f64,
    fmin: f64,
    rectangles: Vec<Rectangle>,
}

impl<'a, F> Direct<F>
where
    F: Fn(ArrayView1<f64>) -> f64 + 'a,
{
    pub fn run(&self) -> (Array1<f64>, f64) {
        let mut rectangles_by_size: Vec<Group> = vec![];
        let mut num_evaluations = 0;
        let (mut xmin, mut fmin) = self.initialize(&mut rectangles_by_size, &mut num_evaluations);
        for _it in 0..self.max_iterations.unwrap_or(usize::MAX) {
            if let Some(max_evaluations) = self.max_evaluations {
                if num_evaluations >= max_evaluations {
                    break;
                }
            }

            let potentially_optimal = self.get_potentially_optimal(&mut rectangles_by_size, fmin);
            if potentially_optimal.is_empty() {
                break;
            }
            for rectangle in potentially_optimal {
                let (split_xmin, split_fmin) =
                    self.split(rectangle, &mut rectangles_by_size, &mut num_evaluations);
                if split_fmin < fmin {
                    xmin = split_xmin;
                    fmin = split_fmin;
                }
            }
        }
        (self.denormalize_point(xmin), fmin)
    }

    fn initialize(
        &self,
        rectangles_by_size: &mut Vec<Group>,
        num_evaluations: &mut usize,
    ) -> (Array1<f64>, f64) {
        let dimensions = self.bounds.len();
        let center = Array1::from_elem(dimensions, 0.5);
        let bound_ranges = Array1::ones(dimensions);

        let center_eval = (self.function)(self.denormalize_point(center.clone()).view());
        *num_evaluations += 1;
        let rectangle = Rectangle {
            center,
            bound_ranges,
            size: (dimensions as f64 * 0.5_f64.powi(2)).sqrt(),
            fmin: center_eval,
        };

        self.split(rectangle, rectangles_by_size, num_evaluations)
    }

    fn get_potentially_optimal(
        &self,
        rectangles_by_size: &mut Vec<Group>,
        fmin: f64,
    ) -> Vec<Rectangle> {
        let fmin_is_zero = fmin.abs() < f64::EPSILON;

        let mut potentially_optimal_group_indices = vec![];

        for (
            i,
            Group {
                size: group_size,
                fmin: group_fmin,
                ..
            },
        ) in rectangles_by_size.iter().enumerate()
        {
            let minimum_larger_diff = rectangles_by_size[i + 1..]
                .iter()
                .map(
                    |Group {
                         size: larger_group_size,
                         fmin: larger_group_fmin,
                         ..
                     }| {
                        (larger_group_fmin - group_fmin) / (larger_group_size - group_size)
                    },
                )
                .min_by(|a, b| a.partial_cmp(&b).unwrap());
            let maximum_smaller_diff = rectangles_by_size[..i]
                .iter()
                .map(
                    |Group {
                         size: smaller_group_size,
                         fmin: smaller_group_fmin,
                         ..
                     }| {
                        (group_fmin - smaller_group_fmin) / (group_size - smaller_group_size)
                    },
                )
                .max_by(|a, b| a.partial_cmp(&b).unwrap());

            let is_potentially_optimal = if let Some(minimum_larger_diff) = minimum_larger_diff {
                // Lemma 3.3 (7)
                let lemma_7_satisfied = if let Some(maximum_smaller_diff) = maximum_smaller_diff {
                    minimum_larger_diff > 0. && maximum_smaller_diff <= minimum_larger_diff
                } else {
                    true
                };
                let lemma_8_or_9_satisfied = if !fmin_is_zero {
                    // Lemma 3.3 (8)
                    self.epsilon
                        <= (fmin - group_fmin) / fmin.abs()
                            + group_size / fmin.abs() * minimum_larger_diff
                } else {
                    // Lemma 3.3 (9)
                    *group_fmin <= group_size * minimum_larger_diff
                };

                lemma_7_satisfied && lemma_8_or_9_satisfied
            } else {
                true
            };

            if is_potentially_optimal {
                potentially_optimal_group_indices.push(i);
            }
        }

        let mut potentially_optimal = vec![];
        for i in potentially_optimal_group_indices.into_iter().rev() {
            let mut group = &mut rectangles_by_size[i];
            // Lemma 3.3 (6)
            let potentially_optimal_rectangle_indices = group
                .rectangles
                .iter()
                .enumerate()
                .filter(|(_, r)| (r.fmin - group.fmin).abs() < f64::EPSILON)
                .map(|(i, _)| i)
                .collect::<Vec<_>>();

            potentially_optimal.reserve(potentially_optimal_rectangle_indices.len());
            for j in potentially_optimal_rectangle_indices.into_iter().rev() {
                potentially_optimal.push(group.rectangles.remove(j));
            }
            if group.rectangles.is_empty() {
                drop(group);
                rectangles_by_size.remove(i);
            } else {
                group.fmin = group
                    .rectangles
                    .iter()
                    .map(|r| r.fmin)
                    .min_by(|a, b| a.partial_cmp(&b).unwrap())
                    .unwrap();
            }
        }

        potentially_optimal
    }

    fn split(
        &self,
        rectangle: Rectangle,
        rectangles: &mut Vec<Group>,
        num_evaluations: &mut usize,
    ) -> (Array1<f64>, f64) {
        let dimensions = rectangle.bound_ranges.len();

        let max_bound_range = rectangle
            .bound_ranges
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let indices = rectangle
            .bound_ranges
            .indexed_iter()
            .filter(|(_, range)| (max_bound_range - *range).abs() < f64::EPSILON)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        // indices x dimensions
        let mut δ_e = Array2::zeros((indices.len(), dimensions));
        for (i, dim) in indices.iter().enumerate() {
            δ_e[[i, *dim]] = rectangle.bound_ranges[*dim] / 3.;
        }

        // indices x 2 x dimensions
        let c_δ_e = stack![Axis(1), &rectangle.center - &δ_e, &rectangle.center + &δ_e];

        // evaluate f for each c +/- δ_e
        // indices x 2
        let f_c_δ_e = c_δ_e.map_axis(Axis(2), |x| {
            (self.function)(self.denormalize_point(x.to_owned()).view())
        });
        *num_evaluations += f_c_δ_e.len();

        let (f_c_δ_e_min_index, f_c_δ_e_min_value) = f_c_δ_e
            .indexed_iter()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        let (split_xmin, split_fmin) = if *f_c_δ_e_min_value < rectangle.fmin {
            (
                c_δ_e
                    .slice(s![f_c_δ_e_min_index.0, f_c_δ_e_min_index.1, ..])
                    .to_owned(),
                *f_c_δ_e_min_value,
            )
        } else {
            (rectangle.center.clone(), rectangle.fmin)
        };

        // for each j, find wj = min(c - δ_ej, c + δ_ej)
        // indices
        let w_values = f_c_δ_e.map_axis(Axis(1), |x| {
            *x.iter().min_by(|a, b| a.partial_cmp(&b).unwrap()).unwrap()
        });

        // Divide starting with the dimension with the smallest (best) wj
        let mut indices_that_sort_w = (0..w_values.len()).collect::<Vec<_>>();
        indices_that_sort_w.sort_by(|i, j| w_values[*i].partial_cmp(&w_values[*j]).unwrap());

        // Each split divides into 3 rectangles
        // Because we may have multiple splits, we keep prev_rectangle for splitting along subsequent wj
        let mut prev_rectangle = rectangle;
        // The size shrinks after each iteration so we only need to search below a previously found size
        let mut binary_search_upper_bound = rectangles.len();

        for wj_index in indices_that_sort_w {
            let dim = indices[wj_index];
            let mut bound_ranges = prev_rectangle.bound_ranges;
            bound_ranges[dim] /= 3.;
            kbn_summation! {
                for range in &bound_ranges => {
                    size_squared += (range / 2.).powi(2);
                }
            }
            let size = size_squared.sqrt();

            // left, right
            match rectangles[..binary_search_upper_bound]
                .binary_search_by(|g| g.size.partial_cmp(&size).unwrap())
            {
                Ok(i) => {
                    let group = &mut rectangles[i];
                    group.fmin = group.fmin.min(w_values[wj_index]);
                    group.rectangles.reserve(2);
                    for k in 0..1 {
                        group.rectangles.push(Rectangle {
                            bound_ranges: bound_ranges.clone(),
                            size,
                            center: c_δ_e.slice(s![wj_index, k, ..]).to_owned(),
                            fmin: f_c_δ_e[[wj_index, k]],
                        });
                    }
                    binary_search_upper_bound = i + 1;
                }
                Err(i) => {
                    rectangles.insert(
                        i,
                        Group {
                            size,
                            fmin: w_values[wj_index],
                            rectangles: vec![
                                Rectangle {
                                    bound_ranges: bound_ranges.clone(),
                                    size,
                                    center: c_δ_e.slice(s![wj_index, 0, ..]).to_owned(),
                                    fmin: f_c_δ_e[[wj_index, 0]],
                                },
                                Rectangle {
                                    bound_ranges: bound_ranges.clone(),
                                    size,
                                    center: c_δ_e.slice(s![wj_index, 1, ..]).to_owned(),
                                    fmin: f_c_δ_e[[wj_index, 1]],
                                },
                            ],
                        },
                    );
                    binary_search_upper_bound = i + 1;
                }
            }

            // center
            prev_rectangle = Rectangle {
                bound_ranges,
                size,
                center: prev_rectangle.center,
                fmin: prev_rectangle.fmin,
            };
        }

        match rectangles[..binary_search_upper_bound]
            .binary_search_by(|g| g.size.partial_cmp(&prev_rectangle.size).unwrap())
        {
            Ok(i) => {
                let group = &mut rectangles[i];
                group.fmin = group.fmin.min(prev_rectangle.fmin);
                group.rectangles.push(prev_rectangle);
            }
            Err(i) => {
                rectangles.insert(
                    i,
                    Group {
                        size: prev_rectangle.size,
                        fmin: prev_rectangle.fmin,
                        rectangles: vec![prev_rectangle],
                    },
                );
            }
        }

        (split_xmin, split_fmin)
    }

    fn denormalize_point(&self, mut hypercube_point: Array1<f64>) -> Array1<f64> {
        azip!(
            (x in &mut hypercube_point, bound in &self.bounds) *x = *x * (bound[1] - bound[0]) + bound[0]
        );
        hypercube_point
    }
}

#[cfg(test)]
mod test {
    use lyon_geom::euclid::{UnknownUnit, Vector3D};
    use ndarray::{array, azip, Array, Array1};

    use crate::{direct::Direct, ColorModel};

    #[test]
    fn test_direct() {
        let direct = Direct {
            epsilon: 1e-4,
            max_evaluations: Some(1000),
            max_iterations: None,
            initial: Array::zeros(2),
            bounds: Array::from_elem(2, [-10., 10.]),
            function: |val| val[0].powi(2) + val[1].powi(2),
        };
        assert_eq!(direct.run().1, 0.);
    }

    #[test]
    fn test_direct_real() {
        // abL
        let implements: Array1<Vector3D<f64, UnknownUnit>> = array![
            Vector3D::from((-12.33001605954215, -45.54515542156117, 44.2098529479848)),
            Vector3D::from((27.880276413952384, -45.45097702564241, 79.59139231597462)),
            Vector3D::from((0.0, 0.0, 100.0)),
            Vector3D::from((25.872063973881424, -58.18583421858581, 77.27752311788944)),
            Vector3D::from((-26.443894809510233, 42.307075530106964, 52.73969688418173)),
            Vector3D::from((66.72215124694603, 8.94553594498204, 50.895965946274)),
            Vector3D::from((86.860821880907, -69.34347889122935, 46.303948069777704)),
            Vector3D::from((21.03625782707445, -63.798798964168235, 67.01735205659284)),
            Vector3D::from((-35.98529688090144, 11.606079999533165, 51.30132332650257)),
            Vector3D::from((62.596792812655295, 33.336563699816914, 55.46042775958594)),
        ];
        // hue, chroma, darkness
        let desired = [1.4826900028611403, 5.177699004088122, 0.27727267822882595];
        let direct = Direct {
            epsilon: 1e-4,
            max_evaluations: Some(1_000_000),
            max_iterations: None,
            initial: Array::zeros(implements.len()),
            bounds: Array::from_elem(implements.len(), [0., 1.]),
            function: |param| {
                let mut weighted_vector = Vector3D::zero();
                azip! {
                    (p in &param, i in &implements) {
                        weighted_vector += *i * *p;
                    }
                }
                // Convert back to cylindrical model (hue, chroma, darkness)
                let actual = [
                    weighted_vector.y.atan2(weighted_vector.x),
                    weighted_vector.to_2d().length(),
                    weighted_vector.z,
                ];
                ColorModel::Cielab.cylindrical_diff(actual, desired)
            },
        };
        let (res, cost) = direct.run();
        let weighted_vector = implements
            .iter()
            .zip(res.iter())
            .map(|(p, x)| *p * *x)
            .sum::<Vector3D<f64, _>>();
        // Convert back to cylindrical model (hue, chroma, darkness)
        let actual = [
            weighted_vector.y.atan2(weighted_vector.x),
            weighted_vector.to_2d().length(),
            weighted_vector.z,
        ];
        dbg!(
            cost,
            &res,
            &actual,
            ColorModel::Cielab.cylindrical_diff(actual, desired)
        );
        assert!(cost <= 4.0, "ciede2000 less than 4");
    }
}
