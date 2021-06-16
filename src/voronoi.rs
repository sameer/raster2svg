use std::fmt::Debug;

use lyon_geom::{point, vector, Line, LineSegment, Point, Vector};
use ndarray::ArrayView2;
use num_traits::{FromPrimitive, PrimInt, Signed};

use crate::abs_distance_squared;

/// Given a set of sites in a bounding box from (0, 0) to (width, height),
/// return the assignment of coordinates in that box to their nearest neighbor
/// using the Jump Flooding Algorithm.
///
/// https://www.comp.nus.edu.sg/~tants/jfa/i3d06.pdf
pub fn jump_flooding_voronoi<T: PrimInt + Signed + FromPrimitive + Debug>(
    sites: &[[T; 2]],
    width: usize,
    height: usize,
) -> Vec<Vec<usize>> {
    if sites.is_empty() {
        return vec![];
    }
    // use usize::MAX to represent colorless cells
    let mut grid = vec![vec![usize::MAX; width]; height];
    sites.iter().enumerate().for_each(|(color, site)| {
        grid[site[1].to_usize().unwrap()][site[0].to_usize().unwrap()] = color;
    });

    let mut round_step = (width.max(height))
        .checked_next_power_of_two()
        .map(|x| x / 2)
        .unwrap_or_else(|| (width.max(height) / 2).next_power_of_two());
    while round_step != 0 {
        for y_dir in -1..=1 {
            let y_range = if y_dir == -1 { round_step } else { 0 }..if y_dir == 1 {
                height.saturating_sub(round_step)
            } else {
                height
            };
            for x_dir in -1..=1 {
                let x_range = if x_dir == -1 { round_step } else { 0 }..if x_dir == 1 {
                    width.saturating_sub(round_step)
                } else {
                    width
                };
                for j in y_range.clone() {
                    let y = match y_dir {
                        -1 => j - round_step,
                        0 => j,
                        1 => j + round_step,
                        _ => unreachable!(),
                    };
                    for i in x_range.clone() {
                        let x = match x_dir {
                            -1 => i - 1,
                            0 => i,
                            1 => i + round_step,
                            _ => unreachable!(),
                        };
                        let new = grid[y][x];
                        if new != usize::MAX {
                            let current = grid[j][i];
                            let here = [T::from_usize(i).unwrap(), T::from_usize(j).unwrap()];
                            if current == usize::MAX
                                || abs_distance_squared(here, sites[new])
                                    < abs_distance_squared(here, sites[current])
                            {
                                grid[j][i] = new;
                            }
                        }
                    }
                }
            }
        }

        round_step /= 2;
    }

    grid
}

pub fn colors_to_assignments<T: PrimInt + FromPrimitive>(
    sites: &[[T; 2]],
    grid: &[Vec<usize>],
) -> Vec<Vec<[T; 2]>> {
    let expected_assignment_capacity =
        grid.len() * grid.first().map(|first| first.len()).unwrap_or(0) / sites.len();
    let mut sites_to_points =
        vec![Vec::<[T; 2]>::with_capacity(expected_assignment_capacity); sites.len()];
    for j in 0..grid.len() {
        for i in 0..grid[j].len() {
            sites_to_points[grid[j][i]]
                .push([T::from_usize(i).unwrap(), T::from_usize(j).unwrap()]);
        }
    }
    sites_to_points
}

#[derive(Default)]
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

#[derive(Default)]
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

    if moments.density != 0.0 {
        let centroid = point(
            (moments.x / moments.density) as f64,
            (moments.y / moments.density) as f64,
        );
        cell_properties.centroid = Some(centroid);

        let x = moments.xx / moments.density - centroid.x.powi(2);
        let y = moments.xy / moments.density - centroid.x * centroid.y;
        let z = moments.yy / moments.density - centroid.y.powi(2);
        let phi = 0.5 * (2.0 * y).atan2(x - z);
        let phi_vector = vector(phi.cos(), phi.sin());
        cell_properties.phi_vector = Some(phi_vector);

        if points.len() >= 3 {
            let hull = crate::hull::convex_hull(points);

            let edge_vectors = hull
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

            assert_eq!(
                edge_vectors.len(),
                2,
                "It should be impossible for a line to intersect a polygon at more than two edges: {:?} {:?} {:?}",
                hull,
                centroid,
                phi_vector,
            );

            cell_properties.hull = Some(hull);

            if let [left, right] = edge_vectors.as_slice() {
                cell_properties.phi_oriented_segment_through_centroid = Some(LineSegment {
                    from: left.to,
                    to: right.to,
                });
            }
        }

        if cell_properties
            .phi_oriented_segment_through_centroid
            .is_none()
        {
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
            [0, HEIGHT - 1],
            [WIDTH - 1, 0],
            [WIDTH - 1, HEIGHT - 1],
            [WIDTH / 2, HEIGHT / 2],
        ];
        let assignments = jump_flooding_voronoi(&sites, WIDTH, HEIGHT);
        for j in 0..HEIGHT {
            for i in 0..WIDTH {
                let min_distance = sites
                    .iter()
                    .map(|site| abs_distance_squared(*site, [i, j]))
                    .min()
                    .unwrap();
                let actual_distance = abs_distance_squared(sites[assignments[j][i]], [i, j]);

                // Don't check the assigned site because of distance ties
                assert_eq!(min_distance, actual_distance);
            }
        }
    }
}
