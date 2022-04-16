use std::fmt::Debug;

use log::warn;
use lyon_geom::{point, vector, Angle, Line, LineSegment, Point, Vector};
use ndarray::{par_azip, prelude::*};
use num_traits::{FromPrimitive, PrimInt, Signed};

use crate::{abs_distance_squared, get_slice_info_for_offset};

mod hull;

/// An arbitrary Voronoi site with 0, 1, or 2 dimensions.
pub trait Site<T: PrimInt + FromPrimitive + Debug> {
    fn dist(&self, point: [T; 2]) -> f64;
    fn seeds(&self, width: usize, height: usize) -> Vec<[T; 2]>;
}

/// 0D site (point)
impl<T> Site<T> for [T; 2]
where
    T: PrimInt + Signed + FromPrimitive + Debug,
{
    fn dist(&self, point: [T; 2]) -> f64 {
        abs_distance_squared(*self, point).to_f64().unwrap()
    }
    fn seeds(&self, _width: usize, _height: usize) -> Vec<[T; 2]> {
        vec![*self]
    }
}

/// 1D site (line segment)
impl<T> Site<T> for LineSegment<f64>
where
    T: PrimInt + FromPrimitive + Debug,
{
    fn dist(&self, p: [T; 2]) -> f64 {
        let p = point(p[0].to_f64().unwrap(), p[1].to_f64().unwrap());
        let square_length = self.to_vector().square_length();
        if square_length.abs() <= f64::EPSILON {
            (p - self.from).square_length();
        }

        let t = ((p - self.from).dot(self.to - self.from) / square_length).clamp(0., 1.);

        (p - self.sample(t)).square_length()
    }

    fn seeds(&self, width: usize, height: usize) -> Vec<[T; 2]> {
        let mut seeds = vec![];

        let width = (width - 1) as f64;
        let height = (height - 1) as f64;

        {
            let start_x = self.from.x.min(self.to.x).floor().clamp(0., width);
            let end_x = self.from.x.max(self.to.x).ceil().clamp(0., width);
            let mut x = start_x;
            while x < end_x {
                let y = self.solve_y_for_x(x).clamp(0., height);
                seeds.push([T::from_f64(x).unwrap(), T::from_f64(y.round()).unwrap()]);
                x += 1.;
            }
        }
        {
            let start_y = self.from.y.min(self.to.y).floor().clamp(0., height);
            let end_y = self.from.y.max(self.to.y).ceil().clamp(0., height);
            let mut y = start_y;
            while y < end_y {
                let x = self.solve_x_for_y(y).clamp(0., width);
                seeds.push([T::from_f64(x).unwrap(), T::from_f64(y.round()).unwrap()]);
                y += 1.;
            }
        }

        seeds
    }
}

/// Given a set of sites in a bounding box from (0, 0) to (width, height),
/// return the assignment of coordinates in that box to their nearest neighbor
/// using the Jump Flooding Algorithm.
///
/// <https://www.comp.nus.edu.sg/~tants/jfa/i3d06.pdf>
pub fn jump_flooding_voronoi<S: Site<T> + Send + Sync, T: PrimInt + FromPrimitive + Debug>(
    sites: &[S],
    width: usize,
    height: usize,
) -> Array2<usize> {
    // use usize::MAX to represent colorless cells
    let mut grid = Array2::from_elem((width, height), usize::MAX);

    if sites.is_empty() {
        return grid;
    }
    sites.iter().enumerate().for_each(|(color, site)| {
        for seed in site.seeds(width, height) {
            grid[[seed[0].to_usize().unwrap(), seed[1].to_usize().unwrap()]] = color;
        }
    });

    let positions = Array::from_iter((0..width).flat_map(|x| (0..height).map(move |y| [x, y])))
        .into_shape((width, height))
        .unwrap();

    let mut scratchpad = grid.clone();

    let mut round_step = (width.max(height))
        .checked_next_power_of_two()
        .map(|x| x / 2)
        .unwrap_or_else(|| (width.max(height) / 2).next_power_of_two());

    while round_step != 0 {
        for y_dir in -1..=1 {
            for x_dir in -1..=1 {
                let center_slice_info = get_slice_info_for_offset(
                    -x_dir * round_step as i32,
                    -y_dir * round_step as i32,
                );
                let kernel_slice_info =
                    get_slice_info_for_offset(x_dir * round_step as i32, y_dir * round_step as i32);
                par_azip! {
                    (dest in scratchpad.slice_mut(center_slice_info), sample in grid.slice(kernel_slice_info), here in positions.slice(center_slice_info)) {
                        let here = [T::from_usize(here[0]).unwrap(), T::from_usize(here[1]).unwrap()];
                        if *sample != usize::MAX && (*dest == usize::MAX || sites[*sample].dist(here) < sites[*dest].dist(here)) {
                            *dest = *sample;
                        }
                    }
                };
                grid.assign(&scratchpad);
            }
        }

        round_step /= 2;
    }

    grid
}

pub fn colors_to_assignments<S: Site<T>, T: PrimInt + FromPrimitive + Debug>(
    sites: &[S],
    grid: ArrayView2<usize>,
) -> Vec<Vec<[T; 2]>> {
    if sites.is_empty() {
        return vec![];
    }
    let expected_assignment_capacity = grid.len() / sites.len();
    let mut sites_to_points =
        vec![Vec::<[T; 2]>::with_capacity(expected_assignment_capacity); sites.len()];
    grid.indexed_iter().for_each(|((i, j), site)| {
        sites_to_points[*site].push([T::from_usize(i).unwrap(), T::from_usize(j).unwrap()])
    });
    sites_to_points
}

#[derive(Default, Debug)]
pub struct Moments {
    pub density: f64,
    pub x: f64,
    pub y: f64,
    pub xx: f64,
    pub xy: f64,
    pub yy: f64,
}

fn calculate_moments<T: PrimInt>(image: ArrayView2<f64>, points: &[[T; 2]]) -> Moments {
    let mut moments = Moments::default();
    for point in points {
        let x = point[0].to_usize().unwrap();
        let y = point[1].to_usize().unwrap();
        let density = image[[x, y]];
        let x = x as f64;
        let y = y as f64;
        moments.density += density;
        moments.x += x * density;
        moments.y += y * density;
        moments.xx += x.powi(2) * density;
        moments.xy += x * y * density;
        moments.yy += y.powi(2) * density;
    }
    moments
}

#[derive(Default, Debug)]
pub struct CellProperties<T: PrimInt + Debug + Default> {
    pub moments: Moments,
    pub centroid: Option<Point<f64>>,
    pub phi_vector: Option<Vector<f64>>,
    pub hull: Option<Vec<[T; 2]>>,
    pub phi_oriented_segment_through_centroid: Option<LineSegment<f64>>,
}

pub fn calculate_cell_properties<T: PrimInt + Debug + Default>(
    image: ArrayView2<f64>,
    points: &[[T; 2]],
) -> CellProperties<T> {
    let moments = calculate_moments(image, points);

    let mut cell_properties = CellProperties::default();

    if moments.density > f64::EPSILON {
        let centroid = point(
            (moments.x / moments.density) as f64,
            (moments.y / moments.density) as f64,
        );
        cell_properties.centroid = Some(centroid);

        let x = moments.xx / moments.density - (moments.x / moments.density).powi(2);
        let y = moments.xy / moments.density - (moments.x * moments.y / moments.density.powi(2));
        let z = moments.yy / moments.density - (moments.y / moments.density).powi(2);
        let phi = Angle::radians(0.5 * (2.0 * y).atan2(x - z));
        let (sin, cos) = phi.sin_cos();
        let phi_vector = vector(cos, sin);
        cell_properties.phi_vector = Some(phi_vector);

        const HULL_EPSILON: f64 = 1E-4;

        if points.len() >= 3 {
            let hull = hull::convex_hull(points);
            // Hull may not be valid if points were collinear, or the centroid may lie directly on a vertex
            if hull.len() >= 3
                && !hull.iter().any(|vertex| {
                    (centroid.x - vertex[0].to_f64().unwrap()).abs() < HULL_EPSILON
                        && (centroid.y - vertex[1].to_f64().unwrap()).abs() < HULL_EPSILON
                })
                && !hull
                    .iter()
                    .zip(hull.iter().skip(1).chain(hull.iter().take(1)))
                    .any(|(from, to)| {
                        let edge = LineSegment {
                            from: point(from[0].to_f64().unwrap(), from[1].to_f64().unwrap()),
                            to: point(to[0].to_f64().unwrap(), to[1].to_f64().unwrap()),
                        };
                        let x = edge.solve_x_for_y(centroid.y);
                        let y = edge.solve_y_for_x(centroid.x);
                        (x - centroid.x).abs() < HULL_EPSILON
                            && (y - centroid.y).abs() < HULL_EPSILON
                    })
            {
                let mut edge_vectors = hull
                    .iter()
                    .zip(hull.iter().skip(1).chain(hull.iter().take(1)))
                    .filter_map(|(from, to)| {
                        let edge = LineSegment {
                            from: point(from[0].to_f64().unwrap(), from[1].to_f64().unwrap()),
                            to: point(to[0].to_f64().unwrap(), to[1].to_f64().unwrap()),
                        };

                        edge.line_intersection(&Line {
                            point: centroid,
                            vector: phi_vector,
                        })
                        .map(|intersection| LineSegment {
                            from: centroid,
                            to: intersection,
                        })
                    })
                    .collect::<Vec<_>>();

                // Resolves the issue of hull vertex intersections and floating point inaccuracy
                edge_vectors.sort_by(|a, b| {
                    a.to.x
                        .partial_cmp(&b.to.x)
                        .unwrap()
                        .then(a.to.y.partial_cmp(&b.to.y).unwrap())
                });
                edge_vectors.dedup_by(|a, b| (a.to - b.to).length() < HULL_EPSILON);

                if edge_vectors.len() != 2 {
                    warn!("It should be impossible for this line to intersect the hull at {} edges: {:?} {:?} {:?} {:?} {:?}", edge_vectors.len(), &edge_vectors, &hull, centroid, phi_vector, moments);
                } else if let [left, right] = edge_vectors.as_slice() {
                    cell_properties.phi_oriented_segment_through_centroid = Some(LineSegment {
                        from: left.to,
                        to: right.to,
                    });
                }
            }

            cell_properties.hull = Some(hull);
        } else {
            let radius = (points.len() as f64 / std::f64::consts::PI).sqrt();
            cell_properties.phi_oriented_segment_through_centroid = Some(LineSegment {
                from: centroid + phi_vector * radius,
                to: centroid - phi_vector * radius,
            });
        };
    }
    cell_properties.moments = moments;

    cell_properties
}

#[cfg(test)]
mod tests {
    use super::jump_flooding_voronoi;
    use crate::abs_distance_squared;

    #[test]
    fn test_jump_flooding_voronoi() {
        const WIDTH: usize = 256;
        const HEIGHT: usize = 256;
        let sites = [
            [0, 0],
            [0, HEIGHT as i64 - 1],
            [WIDTH as i64 - 1, 0],
            [WIDTH as i64 - 1, HEIGHT as i64 - 1],
            [WIDTH as i64 / 2, HEIGHT as i64 / 2],
        ];
        let assignments = jump_flooding_voronoi(&sites, WIDTH, HEIGHT);
        for j in 0..HEIGHT {
            for i in 0..WIDTH {
                let min_distance = sites
                    .iter()
                    .map(|site| abs_distance_squared(*site, [i as i64, j as i64]))
                    .min()
                    .unwrap();
                let actual_distance =
                    abs_distance_squared(sites[assignments[[i, j]]], [i as i64, j as i64]);

                // Don't check the assigned site because of distance ties
                assert_eq!(min_distance, actual_distance);
            }
        }
    }
}
