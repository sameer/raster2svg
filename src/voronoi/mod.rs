use std::fmt::Debug;
use std::hash::Hash;
use std::io::Write;

use crate::kbn_summation;
use log::{info, warn};
use lyon_geom::{euclid::Vector2D, point, Angle, Line, LineSegment, Point, Vector};
use ndarray::{par_azip, prelude::*};
use num_traits::{FromPrimitive, PrimInt, Signed};
use rustc_hash::FxHashSet as HashSet;

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
    T: PrimInt + FromPrimitive + Debug + Hash,
{
    /// Straight line distance from p to the closest point on the line
    fn dist(&self, p: [T; 2]) -> f64 {
        let p = point(p[0].to_f64().unwrap(), p[1].to_f64().unwrap());
        let square_length = self.to_vector().square_length();
        if square_length.abs() <= f64::EPSILON {
            (p - self.from).square_length();
        }

        let t = ((p - self.from).dot(self.to - self.from) / square_length).clamp(0., 1.);

        (p - self.sample(t)).square_length()
    }

    /// All integer points close to the line
    fn seeds(&self, width: usize, height: usize) -> Vec<[T; 2]> {
        let mut seeds = HashSet::default();

        let width = (width - 1) as f64;
        let height = (height - 1) as f64;

        {
            let start_x = self.from.x.min(self.to.x).floor().clamp(0., width);
            let end_x = self.from.x.max(self.to.x).ceil().clamp(0., width);
            let mut x = start_x;
            while x < end_x {
                let y = self.solve_y_for_x(x).round().clamp(0., height);
                seeds.insert([T::from_f64(x).unwrap(), T::from_f64(y).unwrap()]);
                x += 1.;
            }
        }
        {
            let start_y = self.from.y.min(self.to.y).floor().clamp(0., height);
            let end_y = self.from.y.max(self.to.y).ceil().clamp(0., height);
            let mut y = start_y;
            while y < end_y {
                let x = self.solve_x_for_y(y).round().clamp(0., width);
                seeds.insert([T::from_f64(x).unwrap(), T::from_f64(y).unwrap()]);
                y += 1.;
            }
        }

        seeds.into_iter().collect()
    }
}

/// Given a set of sites in a bounding box from (0, 0) to (width, height),
/// return the assignment of coordinates in that box to their nearest neighbor
/// using the Jump Flooding Algorithm.
///
/// Colorless cells will be usize::MAX
///
/// Specifically, this is described in <https://www.comp.nus.edu.sg/~tants/jfa/i3d06.pdf>
pub fn jump_flooding_voronoi<
    S: Site<T> + Send + Sync,
    T: PrimInt + FromPrimitive + Debug + Send + Sync,
>(
    sites: &[S],
    width: usize,
    height: usize,
) -> Array2<usize> {
    // use usize::MAX to represent colorless cells
    let mut grid = Array2::from_elem((width, height), usize::MAX);

    if sites.is_empty() {
        return grid;
    }

    // Prime JFA with seeds
    sites.iter().enumerate().for_each(|(color, site)| {
        for seed in site.seeds(width, height) {
            grid[[seed[0].to_usize().unwrap(), seed[1].to_usize().unwrap()]] = color;
        }
    });

    // Needed to parallelize JFA
    let positions = Array::from_iter((0..width).flat_map(|x| {
        (0..height).map(move |y| [T::from_usize(x).unwrap(), T::from_usize(y).unwrap()])
    }))
    .into_shape((width, height))
    .unwrap();

    let mut scratchpad = grid.clone();

    // First round is of size n/2 where n is a power of 2
    let mut round_step = (width.max(height))
        .checked_next_power_of_two()
        .map(|x| x / 2)
        .unwrap_or_else(|| (width.max(height) / 2).next_power_of_two());

    while round_step != 0 {
        // Each grid point passes its contents on to (x+i, y+j) where i,j in {-round_step, 0, round_step}

        // This might look a bit weird... but it's JFA in parallel.
        // For each i,j it runs the jump on the entire image at once
        //
        // This works because JFA is linear:
        // If x,y will propagate to x+k,y+k, then x+1,y+1 will propagate to x+1+k,y+1+k
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
                        *dest =if *dest == usize::MAX {
                            *sample
                        } else if *sample == usize::MAX {
                            *dest
                        } else if sites[*sample].dist(*here) < sites[*dest].dist(*here) {
                            *sample
                        } else {
                            *dest
                        };
                    }
                };
                grid.assign(&scratchpad);
            }
        }

        round_step /= 2;
    }

    grid
}

/// Converts a JFA-assigned color grid into a list of points assigned to each site
pub fn colors_to_assignments<S: Site<T>, T: PrimInt + FromPrimitive + Debug>(
    sites: &[S],
    grid: ArrayView2<usize>,
) -> Vec<Vec<[T; 2]>> {
    if sites.is_empty() {
        return vec![];
    }
    let expected_assignment_capacity = grid.len() / sites.len();
    let mut sites_to_points = vec![Vec::with_capacity(expected_assignment_capacity); sites.len()];
    grid.indexed_iter().for_each(|((i, j), site)| {
        sites_to_points[*site].push([T::from_usize(i).unwrap(), T::from_usize(j).unwrap()])
    });
    sites_to_points
}

/// First and second order moments of a Voronoi cell
#[derive(Default, Debug)]
pub struct Moments {
    /// Sum of the values of points in the cell
    pub m00: f64,
    /// First order x moment
    pub m10: f64,
    /// First order y moment
    pub m01: f64,
    /// Second order xx central moment
    pub μ20: f64,
    /// Second order yy central moment
    pub μ02: f64,
    /// Second order xy central moment
    pub μ11: f64,
    /// Calculated centroid, may be `NaN`
    centroid: Point<f64>,
}

/// Hiller et al. Section 4 + Appendix B
#[inline]
fn calculate_moments<T: PrimInt>(image: ArrayView2<f64>, points: &[[T; 2]]) -> Moments {
    let mut moments = Moments::default();
    kbn_summation! {
        for [x, y] in points => {
            'loop: {
                let x = x.to_usize().unwrap();
                let y = y.to_usize().unwrap();
                let value = image[[x, y]];
                let x = x as f64;
                let y = y as f64;
            }
            m00 += value;
            m10 += x * value;
            m01 += y * value;
        }
    }
    moments.m00 = m00;
    moments.m10 = m10;
    moments.m01 = m01;

    // Hiller et al. Appendix B mass centroid
    let centroid = point(moments.m10 / moments.m00, moments.m01 / moments.m00);
    moments.centroid = centroid;

    // Hiller et al. Appendix B central moments
    kbn_summation! {
        for [x, y] in points => {
            'loop: {
                let x = x.to_usize().unwrap();
                let y = y.to_usize().unwrap();
                let value = image[[x, y]];
                let x = x as f64;
                let y = y as f64;
            }
            μ20 += (x - centroid.x).powi(2) * value;
            μ02 += (y - centroid.y).powi(2) * value;
            μ11 += (x - centroid.x) * (y - centroid.y) * value;
        }
    }
    moments.μ20 = μ20;
    moments.μ02 = μ02;
    moments.μ11 = μ11;

    moments
}

/// Deussen et al 3.3 Beyond Stippling
#[derive(Default, Debug)]
pub struct CellProperties<T: PrimInt + Debug + Default> {
    pub moments: Moments,
    /// Density-based center of the cell
    pub centroid: Option<Point<f64>>,
    /// Orientation of cell's inertial axis
    pub phi_vector: Option<Vector<f64>>,
    /// Convex hull enclosing the cell
    pub hull: Option<Vec<[T; 2]>>,
    /// Used to determine the splitting direction
    pub phi_oriented_segment_through_centroid: Option<LineSegment<f64>>,
}

pub fn calculate_cell_properties<T: PrimInt + FromPrimitive + Debug + Default>(
    image: ArrayView2<f64>,
    points: &[[T; 2]],
) -> CellProperties<T> {
    let mut cell_properties = CellProperties {
        moments: calculate_moments(image, points),
        ..Default::default()
    };

    let moments = &cell_properties.moments;

    // Any calculation here is pointless
    if moments.m00 <= f64::EPSILON {
        return cell_properties;
    }
    let centroid = moments.centroid;
    cell_properties.centroid = Some(moments.centroid);

    // Hiller et al. Appendix B Equation 5
    let phi = Angle::radians(0.5 * (2.0 * moments.μ11).atan2(moments.μ20 - moments.μ02));
    let phi_vector = Vector2D::from_angle_and_length(phi, 1.);
    cell_properties.phi_vector = Some(phi_vector);

    const HULL_EPSILON: f64 = 1E-8;

    if points.len() >= 3 {
        let hull = hull::convex_hull(points);

        let edges_it = hull
            .iter()
            .zip(hull.iter().skip(1).chain(hull.iter().take(1)))
            .map(|(from, to)| LineSegment {
                from: point(from[0].to_f64().unwrap(), from[1].to_f64().unwrap()),
                to: point(to[0].to_f64().unwrap(), to[1].to_f64().unwrap()),
            });
        // Hull may not be valid if: points were colinear
        if hull.len() >= 3 {
            let phi_line = Line {
                point: centroid,
                vector: phi_vector,
            };
            let mut edges_intersecting_phi_line = edges_it
                .clone()
                .filter_map(|edge| {
                    edge.line_intersection(&phi_line)
                        .map(|intersection| LineSegment {
                            from: centroid,
                            to: intersection,
                        })
                })
                .collect::<Vec<_>>();

            // We may see more than 2 intersections if:
            // * Phi line intersects voronoi cell at a vertex, thus intersecting two edges of its hull
            // * Floating point accuracy problems
            edges_intersecting_phi_line.sort_by(|a, b| {
                a.to.x
                    .partial_cmp(&b.to.x)
                    .unwrap()
                    .then(a.to.y.partial_cmp(&b.to.y).unwrap())
            });
            edges_intersecting_phi_line.dedup_by(|a, b| (a.to - b.to).length() < HULL_EPSILON);

            let phi_oriented_segment_through_centroid = match edges_intersecting_phi_line.as_slice()
            {
                // Degenerate case: centroid on edge with phi line parallel to the edge
                [] => {
                    let overlap = edges_it
                        .clone()
                        .min_by(|edge_a, edge_b| {
                            edge_a
                                .to_line()
                                .distance_to_point(&centroid)
                                .partial_cmp(&edge_b.to_line().distance_to_point(&centroid))
                                .unwrap()
                        })
                        .unwrap();

                    let dist = overlap.to_line().distance_to_point(&centroid);
                    #[cfg(debug_assertions)]
                    if dist > HULL_EPSILON {
                        to_svg(&hull, centroid, phi_vector, &edges_intersecting_phi_line);
                    }
                    overlap
                },
                // Degenerate case: centroid on vertex of cell with phi pointing out of the cell
                [segment] => {
                    debug_assert!(
                        segment.length() < HULL_EPSILON,
                        "Length is expected to be 0 for this degenerate case: {:?}",
                        segment
                    );
                    *segment
                }
                [left, right] => LineSegment {
                    from: left.to,
                    to: right.to,
                },
                other => unreachable!("Should not be possible for a line to intersect a convex polygon at more than 2 edges: {:?}", other),
            };
            cell_properties.phi_oriented_segment_through_centroid =
                Some(phi_oriented_segment_through_centroid);
        }

        cell_properties.hull = Some(hull);
    } else {
        let radius = (points.len() as f64 / std::f64::consts::PI).sqrt();
        cell_properties.phi_oriented_segment_through_centroid = Some(LineSegment {
            from: centroid + phi_vector * radius,
            to: centroid - phi_vector * radius,
        });
    };

    cell_properties
}

fn to_svg<T: PrimInt + Debug>(
    hull: &[[T; 2]],
    centroid: Point<f64>,
    phi_vector: Vector<f64>,
    edges_intersecting_phi_line: &[LineSegment<f64>],
) {
    let mut path = String::new();
    if let Some(point) = hull.first() {
        path += &format!("M{:?},{:?} ", point[0], point[1]);
    }
    if hull.len() >= 2 {
        path += "L";
    }
    for point in hull.iter().skip(1) {
        path += &format!("{:?},{:?} ", point[0], point[1]);
    }
    if hull.len() >= 3 {
        path += "Z";
    }

    let [cx, cy] = centroid.to_array();
    let [p1x, p1y] = (centroid + phi_vector * -10.).to_array();
    let [p2x, p2y] = (centroid + phi_vector * 10.).to_array();
    let mut minx = p1x.min(p2x);
    let mut miny = p1y.min(p2y);
    let mut maxx = p1x.max(p2x);
    let mut maxy = p1y.max(p2y);
    for point in hull {
        maxx = maxx.max(point[0].to_usize().unwrap() as f64);
        maxy = maxy.max(point[1].to_usize().unwrap() as f64);
        minx = minx.min(point[0].to_usize().unwrap() as f64);
        miny = miny.min(point[1].to_usize().unwrap() as f64);
    }
    minx -= 2.;
    miny -= 2.;
    maxx += 2.;
    maxy += 2.;
    let mut f =
        std::fs::File::create(format!("/tmp/foo{},{}.svg", centroid.x, centroid.y)).unwrap();
    info!(
        "This is weird /tmp/foo{},{}.svg {}",
        centroid.x,
        centroid.y,
        edges_intersecting_phi_line.len()
    );
    let mut edges = String::default();
    for LineSegment {
        from: Point {
            x: fromx, y: fromy, ..
        },
        to: Point { x: tox, y: toy, .. },
    } in edges_intersecting_phi_line
    {
        edges += &format!(r#"<path stroke="blue" d="M{fromx},{fromy}L{tox},{toy}"/>"#);
    }
    let width = maxx - minx;
    let height = maxy - miny;
    write!(
        f,
        r#"<?xml version="1.0" encoding="UTF-8"?>
<svg fill-opacity="0" stroke-width="0.5" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}mm" height="{height}mm" viewBox="{minx} {miny} {maxx} {maxy}" version="1.1">
<path stroke="black" d="{path}"/>
<path stroke="green" d="M{p1x},{p1y} L{p2x},{p2y}"/>
{edges}
<circle stroke="red" cx="{cx}" cy="{cy}" r="0.5"/>
</svg>"#,
    ).unwrap();
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
