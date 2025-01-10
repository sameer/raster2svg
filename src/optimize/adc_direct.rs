//! Adaptive Diagonal Curves DIRECT algorithm
//!
//! This approach differs from DIRECT in a few key areas:
//!
//! - Two extrema of each rectangle is sampled, rather than the center
//!     - Extremely important when optima lie at boundaries
//! - Only one largest dimension is chosen when splitting a rectangle
//! - Potentially optimal rectangles are selected using a convex hull approach
//!
//! Differences from the ADC DIRECT paper:
//!
//! - L1 norm for rectangle size instead of L2
//! - Track repeated function evaluations by using rational (fraction) coordinates
//! - When splitting one of several largest dimensions, pick the one that has been split least often
//!
//! <https://arxiv.org/pdf/1103.2056>
//! <https://public.websites.umich.edu/~mdolaboratory/pdf/Jones2020a.pdf>

use std::collections::BinaryHeap;

use ndarray::{azip, Array1, ArrayView1};
use num_rational::Rational64;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rustc_hash::FxHashMap as HashMap;

use crate::voronoi::hull::lower_convex_hull;

pub struct AdcDirect<F>
where
    F: Fn(ArrayView1<f64>) -> f64,
{
    pub function: F,
    pub bounds: Array1<[f64; 2]>,
    pub max_evaluations: Option<usize>,
    pub max_iterations: Option<usize>,
}

#[derive(Debug)]
struct AdcDirectState {
    iterations: usize,
    evaluations: HashMap<Array1<Rational64>, f64>,
    rectangles_by_size: Vec<Group>,
    dimension_split_counters: Vec<usize>,
    xmin: Array1<Rational64>,
    fmin: f64,
}

/// Hyper-rectangle as defined by the DIRECT algorithm.
#[derive(Debug, PartialEq)]
struct Rectangle {
    /// Lower bound
    a: Array1<Rational64>,
    /// Upper bound
    b: Array1<Rational64>,
    /// Where this rectangle lies on the Y-axis of the graph built during [`AdcDirect::split`]
    f_graph_pos: f64,
}

impl Rectangle {
    fn new(a: Array1<Rational64>, b: Array1<Rational64>, f_a: f64, f_b: f64) -> Self {
        Self {
            a,
            b,
            f_graph_pos: (f_a + f_b) / 2.,
        }
    }
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
        self.f_graph_pos
            .partial_cmp(&other.f_graph_pos)
            .unwrap()
            .reverse()
    }
}

/// A set of [Rectangles](Rectangle) with the same size, or area.
#[derive(Debug)]
struct Group {
    size: Rational64,
    rectangles: BinaryHeap<Rectangle>,
}

impl Group {
    fn min_f_graph_pos(&self) -> f64 {
        self.rectangles.peek().expect("non empty").f_graph_pos
    }
}

impl<F> AdcDirect<F>
where
    F: Fn(ArrayView1<f64>) -> f64,
{
    pub fn run(&self) -> (Array1<f64>, f64) {
        let mut state = AdcDirectState {
            iterations: 0,
            evaluations: HashMap::default(),
            rectangles_by_size: vec![],
            dimension_split_counters: vec![0; self.bounds.len()],
            xmin: Array1::zeros(self.bounds.len()),
            fmin: 0.,
        };
        self.initialize(&mut state);

        loop {
            if let Some(max_evaluations) = self.max_evaluations {
                if state.evaluations.len() >= max_evaluations {
                    break;
                }
            }
            if let Some(max_iterations) = self.max_iterations {
                if state.iterations >= max_iterations {
                    break;
                }
            }

            let potentially_optimal = self.extract_potentially_optimal(&mut state);
            if potentially_optimal.is_empty() {
                break;
            }
            for rectangle in potentially_optimal {
                self.split(rectangle, &mut state);
            }
            state.iterations += 1;
        }
        (self.denormalize_point(state.xmin.view()), state.fmin)
    }

    /// Initialize data structures following Section 3.2
    fn initialize(&self, state: &mut AdcDirectState) {
        let dimensions = self.bounds.len();

        let a = Array1::from_elem(dimensions, Rational64::zero());
        let b = Array1::from_elem(dimensions, Rational64::one());
        let f_a = (self.function)(self.denormalize_point(a.view()).view());
        let f_b = (self.function)(self.denormalize_point(b.view()).view());
        state
            .evaluations
            .extend([(a.clone(), f_a), (b.clone(), f_b)]);
        state.xmin = if f_a < f_b { a.clone() } else { b.clone() };
        state.fmin = f_a.min(f_b);

        let rectangle = Rectangle::new(a, b, f_a, f_b);
        self.split(rectangle, state);
    }

    /// Identify and extract potentially optimal rectangles.
    fn extract_potentially_optimal(
        &self,
        AdcDirectState {
            rectangles_by_size,
            fmin,
            ..
        }: &mut AdcDirectState,
    ) -> Vec<Rectangle> {
        let points = [[0., *fmin]]
            .into_iter()
            .chain(
                rectangles_by_size
                    .iter()
                    .map(|g| [g.size.to_f64().unwrap(), g.min_f_graph_pos()]),
            )
            .collect::<Vec<_>>();

        let lower_hull_optimal = lower_convex_hull(&points);

        let mut group_it = rectangles_by_size.iter_mut();
        let mut potentially_optimal = vec![];
        // Ignore first point
        for [group_size, group_min_f_graph_pos] in lower_hull_optimal.iter().skip(1) {
            // Find group and extract potentially optimal
            let group = loop {
                let Some(group) = group_it.next() else {
                    unreachable!("there should always be a matching group in the hull");
                };
                if *group_size == group.size.to_f64().unwrap() {
                    break group;
                }
            };
            while !group.rectangles.is_empty() && group.min_f_graph_pos() == *group_min_f_graph_pos
            {
                potentially_optimal.push(group.rectangles.pop().unwrap());
            }
        }
        rectangles_by_size.retain(|g| !g.rectangles.is_empty());

        potentially_optimal
    }

    /// Split the given [Rectangle].
    fn split(
        &self,
        Rectangle { a, b, .. }: Rectangle,
        AdcDirectState {
            evaluations,
            rectangles_by_size,
            dimension_split_counters,
            xmin,
            fmin,
            ..
        }: &mut AdcDirectState,
    ) {
        // Pick a single longest bound, using split counts for tie-breaking.
        let (largest_dimension, _) = b
            .iter()
            .zip(a.iter())
            .map(|(b_i, a_i)| (b_i - a_i).abs())
            .enumerate()
            .max_by(|(i, x), (j, y)| {
                x.cmp(y).then(
                    dimension_split_counters[*i]
                        .cmp(&dimension_split_counters[*j])
                        .reverse(),
                )
            })
            .unwrap();

        // Update split counters for tie-breaking.
        dimension_split_counters[largest_dimension] += 1;

        const SPLIT_RATIO: Rational64 = Rational64::new_raw(2, 3);
        let mut u = a.clone();
        u[largest_dimension] += (b[largest_dimension] - a[largest_dimension]) * SPLIT_RATIO;

        let mut v = b.clone();
        v[largest_dimension] += (a[largest_dimension] - b[largest_dimension]) * SPLIT_RATIO;

        let f_a = *evaluations.get(&a).unwrap();
        let f_b = *evaluations.get(&b).unwrap();
        let f_u = *evaluations
            .entry(u.clone())
            .or_insert_with(|| (self.function)(self.denormalize_point(u.view()).view()));
        let f_v = *evaluations
            .entry(v.clone())
            .or_insert_with(|| (self.function)(self.denormalize_point(v.view()).view()));

        if f_u < *fmin {
            *xmin = u.clone();
            *fmin = f_u;
        }
        if f_v < *fmin {
            *xmin = v.clone();
            *fmin = f_v;
        }

        // This is actually the L1 norm which speeds up convergence significantly.
        // I suspect this is because the rectangles are more grouped up, forcing more local search.
        let size = b
            .iter()
            .zip(a.iter())
            .map(|(b_i, a_i)| (b_i - a_i).abs())
            .sum::<Rational64>();
        let new_rectangles = [
            Rectangle::new(u.clone(), v.clone(), f_u, f_v),
            Rectangle::new(a, v, f_a, f_v),
            Rectangle::new(u, b, f_u, f_b),
        ];

        match rectangles_by_size.binary_search_by(|g| g.size.cmp(&size)) {
            Ok(i) => rectangles_by_size[i].rectangles.extend(new_rectangles),
            Err(i) => {
                rectangles_by_size.insert(
                    i,
                    Group {
                        size,
                        rectangles: BinaryHeap::from(new_rectangles),
                    },
                )
            }
        }
    }

    /// Convert a point from the hypercube range back into user range.
    fn denormalize_point(&self, hypercube_point: ArrayView1<Rational64>) -> Array1<f64> {
        let mut denormalized = hypercube_point.mapv(|x| x.to_f64().unwrap());
        azip!(
            (x in &mut denormalized, bound in &self.bounds) *x = *x * (bound[1] - bound[0]) + bound[0]
        );
        denormalized
    }
}

#[cfg(test)]
mod test {
    use lyon_geom::euclid::default::Vector3D;
    use ndarray::Array;

    use super::AdcDirect;
    use crate::ColorModel;

    #[test]
    fn test_direct() {
        let direct = AdcDirect {
            function: |val| val[0].powi(2) + val[1].powi(2),
            bounds: Array::from_elem(2, [-10., 10.]),
            max_evaluations: Some(10_000),
            max_iterations: None,
        };
        assert_eq!(direct.run().1, 0.);
    }

    #[test]
    fn test_direct_real() {
        // abL
        let implements: Vec<Vector3D<f64>> = vec![
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
        let model = ColorModel::Cielab;
        // hue, chroma, darkness
        let desired = [1.4826900028611403, 5.177699004088122, 0.27727267822882595];
        let direct = AdcDirect {
            function: model.objective_function(desired, &implements),
            bounds: Array::from_elem(implements.len(), [0., 1.]),
            max_evaluations: Some(10_000),
            max_iterations: None,
        };
        let (res, cost) = direct.run();
        let weighted_vector = implements
            .iter()
            .zip(res.iter())
            .map(|(p, x)| *p * *x)
            .sum::<Vector3D<f64>>();
        // Convert back to cylindrical model (hue, chroma, darkness)
        let actual = [
            weighted_vector.y.atan2(weighted_vector.x),
            weighted_vector.to_2d().length(),
            weighted_vector.z,
        ];
        dbg!(cost, &res, &actual, model.cylindrical_diff(desired, actual));
        assert!(cost <= 4.0, "ciede2000 less than 4");
    }
}
