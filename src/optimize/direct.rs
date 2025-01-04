//! DIRECT algorithm implementation, with added improvements.
//!
//! <http://www2.peq.coppe.ufrj.br/Pessoal/Professores/Arge/COQ897/Naturais/DirectUserGuide.pdf>
//! <https://public.websites.umich.edu/~mdolaboratory/pdf/Jones2020a.pdf>

use std::collections::BinaryHeap;

use ndarray::{azip, s, stack, Array1, Array2, ArrayView1, Axis};

use crate::kbn_summation;

pub struct Direct<F>
where
    F: Fn(ArrayView1<f64>) -> f64,
{
    pub function: F,
    pub bounds: Array1<[f64; 2]>,
    pub max_evaluations: Option<usize>,
    pub max_iterations: Option<usize>,
    pub adapt_epsilon: bool,
}

/// Hyper-rectangle as defined by the DIRECT algorithm.
#[derive(Debug, PartialEq)]
struct Rectangle {
    /// Bounds are represented as their length, rather than the endpoints, which can be derived from the center.
    bound_lengths: Array1<f64>,
    size: f64,
    center: Array1<f64>,
    fmin: f64,
}

impl Eq for Rectangle {}

impl PartialOrd for Rectangle {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// For ergonomics, the order is reversed here instead of with a [std::cmp::Reverse] wrapper.
impl Ord for Rectangle {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.fmin.partial_cmp(&other.fmin).unwrap().reverse()
    }
}

/// A set of [Rectangles](Rectangle) with the same size, or area.
#[derive(Debug)]
struct Group {
    size: f64,
    rectangles: BinaryHeap<Rectangle>,
}

impl Group {
    fn fmin(&self) -> f64 {
        self.rectangles.peek().expect("non empty").fmin
    }
}

impl<F> Direct<F>
where
    F: Fn(ArrayView1<f64>) -> f64,
{
    pub fn run(&self) -> (Array1<f64>, f64) {
        let mut rectangles_by_size: Vec<Group> = vec![];
        let mut epsilon = AdaptiveEpsilon::new(self.adapt_epsilon);
        let mut num_evaluations = 0;
        let mut num_iterations = 0;
        let (mut xmin, mut fmin) = self.initialize(&mut rectangles_by_size, &mut num_evaluations);

        loop {
            if let Some(max_evaluations) = self.max_evaluations {
                if num_evaluations >= max_evaluations {
                    break;
                }
            }
            if let Some(max_iterations) = self.max_iterations {
                if num_iterations >= max_iterations {
                    break;
                }
            }

            let potentially_optimal =
                self.extract_potentially_optimal(&mut rectangles_by_size, fmin, &epsilon);
            if potentially_optimal.is_empty() {
                break;
            }
            for rectangle in potentially_optimal {
                let (split_xmin, split_fmin) =
                    self.split(rectangle, &mut rectangles_by_size, &mut num_evaluations);
                if split_fmin < fmin {
                    xmin = split_xmin;
                    fmin = split_fmin;
                    epsilon.improved(num_iterations);
                } else {
                    epsilon.no_improvement(num_iterations);
                }
            }
            num_iterations += 1;
        }
        (self.denormalize_point(xmin), fmin)
    }

    /// Initialize data structures following Section 3.2
    fn initialize(
        &self,
        rectangles_by_size: &mut Vec<Group>,
        num_evaluations: &mut usize,
    ) -> (Array1<f64>, f64) {
        let dimensions = self.bounds.len();
        let center = Array1::from_elem(dimensions, 0.5);
        let bound_lengths = Array1::ones(dimensions);

        let center_eval = (self.function)(self.denormalize_point(center.clone()).view());
        *num_evaluations += 1;
        let rectangle = Rectangle {
            center,
            bound_lengths,
            size: (dimensions as f64 * 0.5_f64.powi(2)).sqrt(),
            fmin: center_eval,
        };

        self.split(rectangle, rectangles_by_size, num_evaluations)
    }

    /// Identify and extract potentially optimal rectangles.
    fn extract_potentially_optimal(
        &self,
        rectangles_by_size: &mut Vec<Group>,
        fmin: f64,
        epsilon: &AdaptiveEpsilon,
    ) -> Vec<Rectangle> {
        let fmin_is_zero = fmin.abs() < f64::EPSILON;

        let mut potentially_optimal_group_indices = vec![];

        for i in (0..rectangles_by_size.len()).rev() {
            let group = &rectangles_by_size[i];

            // Lemma 3.3 (7) values
            let minimum_larger_diff = rectangles_by_size[i + 1..]
                .iter()
                .map(|larger_group| {
                    (larger_group.fmin() - group.fmin()) / (larger_group.size - group.size)
                })
                .min_by(|a, b| a.partial_cmp(b).unwrap());
            let maximum_smaller_diff = rectangles_by_size[..i]
                .iter()
                .map(|smaller_group| {
                    (group.fmin() - smaller_group.fmin()) / (group.size - smaller_group.size)
                })
                .max_by(|a, b| a.partial_cmp(b).unwrap());

            let is_potentially_optimal = if let Some(minimum_larger_diff) = minimum_larger_diff {
                // Lemma 3.3 (7)
                let lemma_7_satisfied = if let Some(maximum_smaller_diff) = maximum_smaller_diff {
                    minimum_larger_diff > 0. && maximum_smaller_diff <= minimum_larger_diff
                } else {
                    true
                };
                let lemma_8_or_9_satisfied = if !fmin_is_zero {
                    // Lemma 3.3 (8)
                    epsilon.value()
                        <= (fmin - group.fmin()) / fmin.abs()
                            + group.size / fmin.abs() * minimum_larger_diff
                } else {
                    // Lemma 3.3 (9)
                    group.fmin() <= group.size * minimum_larger_diff
                };

                lemma_7_satisfied && lemma_8_or_9_satisfied
            } else {
                true
            };

            if is_potentially_optimal {
                // Lemma 3.3 (6)
                potentially_optimal_group_indices.push(i);
            }
        }

        let mut potentially_optimal = vec![];
        for i in potentially_optimal_group_indices {
            let group = &mut rectangles_by_size[i];
            let group_fmin = group.fmin();
            while let Some(rectangle) = group.rectangles.peek() {
                if (rectangle.fmin - group_fmin).abs() < f64::EPSILON {
                    potentially_optimal.push(group.rectangles.pop().unwrap());
                } else {
                    break;
                }
            }
        }

        // Update data structures after extracting rectangles.
        rectangles_by_size.retain(|g| !g.rectangles.is_empty());

        potentially_optimal
    }

    /// Split the given [Rectangle].
    fn split(
        &self,
        rectangle: Rectangle,
        rectangles: &mut Vec<Group>,
        num_evaluations: &mut usize,
    ) -> (Array1<f64>, f64) {
        let dimensions = rectangle.bound_lengths.len();

        // Find the longest bound.
        let longest_bound_length = rectangle
            .bound_lengths
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        // Split along all bounds that are the same length as the longest bound.
        let indices = rectangle
            .bound_lengths
            .indexed_iter()
            .filter(|(_, len)| (longest_bound_length - *len).abs() < f64::EPSILON)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        // Difference vector for each dimension being split (indices x dimensions)
        let mut δ_e = Array2::zeros((indices.len(), dimensions));
        for (i, dim) in indices.iter().enumerate() {
            δ_e[[i, *dim]] = rectangle.bound_lengths[*dim] / 3.;
        }

        // function inputs (indices x 2 x dimensions)
        let c_δ_e = stack![Axis(1), &rectangle.center - &δ_e, &rectangle.center + &δ_e];

        // evaluate function for each c +/- δ_e (indices x 2)
        let f_c_δ_e = c_δ_e.map_axis(Axis(2), |x| {
            (self.function)(self.denormalize_point(x.to_owned()).view())
        });
        *num_evaluations += f_c_δ_e.len();

        let (f_c_δ_e_min_index, f_c_δ_e_min_value) = f_c_δ_e
            .indexed_iter()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        // The minimum function evaluation found during this split
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
            *x.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
        });

        // Divide starting with the dimension with the smallest (best) wj
        let mut indices_that_sort_w = (0..w_values.len()).collect::<Vec<_>>();
        indices_that_sort_w
            .sort_unstable_by(|i, j| w_values[*i].partial_cmp(&w_values[*j]).unwrap());

        // Each split divides into 3 rectangles.
        // Because we may have multiple splits, we keep prev_rectangle for splitting along subsequent wj.
        let mut prev_rectangle = rectangle;
        // The size shrinks after each iteration so we only need to search below a previously found size.
        let mut binary_search_upper_bound = rectangles.len();

        for wj_index in indices_that_sort_w {
            let dim = indices[wj_index];
            let mut bound_lengths = prev_rectangle.bound_lengths;
            bound_lengths[dim] /= 3.;
            kbn_summation! {
                for len in &bound_lengths => {
                    size_squared += (len / 2.).powi(2);
                }
            }
            let size = size_squared.sqrt();

            // Insert left, right
            let left_and_right = (0..1).map(|k| Rectangle {
                bound_lengths: bound_lengths.clone(),
                size,
                center: c_δ_e.slice(s![wj_index, k, ..]).to_owned(),
                fmin: f_c_δ_e[[wj_index, k]],
            });
            match rectangles[..binary_search_upper_bound]
                .binary_search_by(|g| g.size.partial_cmp(&size).unwrap())
            {
                Ok(i) => {
                    rectangles[i].rectangles.extend(left_and_right);
                    binary_search_upper_bound = i + 1;
                }
                Err(i) => {
                    rectangles.insert(
                        i,
                        Group {
                            size,
                            rectangles: BinaryHeap::from_iter(left_and_right),
                        },
                    );
                    binary_search_upper_bound = i + 1;
                }
            }

            // Keep track of center for further splitting and re-insertion.
            prev_rectangle = Rectangle {
                bound_lengths,
                size,
                center: prev_rectangle.center,
                fmin: prev_rectangle.fmin,
            };
        }

        // Put the center rectangle back in.
        match rectangles[..binary_search_upper_bound]
            .binary_search_by(|g| g.size.partial_cmp(&prev_rectangle.size).unwrap())
        {
            Ok(i) => {
                rectangles[i].rectangles.push(prev_rectangle);
            }
            Err(i) => {
                rectangles.insert(
                    i,
                    Group {
                        size: prev_rectangle.size,
                        rectangles: BinaryHeap::from([prev_rectangle]),
                    },
                );
            }
        }

        (split_xmin, split_fmin)
    }

    /// Convert a point from the hypercube range back into user range.
    fn denormalize_point(&self, mut hypercube_point: Array1<f64>) -> Array1<f64> {
        azip!(
            (x in &mut hypercube_point, bound in &self.bounds) *x = *x * (bound[1] - bound[0]) + bound[0]
        );
        hypercube_point
    }
}

/// DIRECT-restart
struct AdaptiveEpsilon {
    enabled: bool,
    prefer_locality: bool,
    last_improved_iteration: usize,
}

impl AdaptiveEpsilon {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            prefer_locality: true,
            last_improved_iteration: 0,
        }
    }

    fn value(&self) -> f64 {
        if !self.enabled {
            1E-4
        } else {
            match self.prefer_locality {
                true => 0.,
                false => 0.01,
            }
        }
    }

    fn improved(&mut self, iteration: usize) {
        self.last_improved_iteration = iteration;
        self.prefer_locality = true;
    }

    fn no_improvement(&mut self, iteration: usize) {
        let num_consecutive_before_action = match self.prefer_locality {
            true => 5,
            false => 50,
        };

        if iteration - self.last_improved_iteration >= num_consecutive_before_action {
            self.prefer_locality = !self.prefer_locality;
        }
    }
}

#[cfg(test)]
mod test {
    use lyon_geom::euclid::{UnknownUnit, Vector3D};
    use ndarray::{array, azip, Array, Array1};

    use super::Direct;
    use crate::ColorModel;

    #[test]
    fn test_direct() {
        let direct = Direct {
            adapt_epsilon: true,
            max_evaluations: Some(1000),
            max_iterations: None,
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
            adapt_epsilon: true,
            max_evaluations: Some(1_000_000),
            max_iterations: None,
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
                ColorModel::Cielab.cylindrical_diff(desired, actual)
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
            ColorModel::Cielab.cylindrical_diff(desired, actual)
        );
        assert!(cost <= 4.0, "ciede2000 less than 4");
    }
}
