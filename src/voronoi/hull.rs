use std::{
    cmp::PartialOrd,
    ops::{Add, Mul},
};

use num_traits::Zero;

/// Andrew's monotone chain convex hull algorithm
///
/// Points must be sorted by x-coordinate and tie-broken by y-coordinate. Collinear points will be excluded from the hull.
///
/// <https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain>
pub fn convex_hull<T>(points: &[[T; 2]]) -> Vec<[T; 2]>
where
    T: Zero + Mul<Output = T> + Add<Output = T> + Copy + PartialOrd,
{
    if points.len() <= 3 {
        return points.to_vec();
    }

    let mut lower = lower_convex_hull(points);
    lower.pop();

    let mut upper = upper_convex_hull(points);
    upper.pop();

    lower.append(&mut upper);
    lower
}

/// Lower half of [`convex_hull`].
pub fn lower_convex_hull<T>(points: &[[T; 2]]) -> Vec<[T; 2]>
where
    T: Zero + Mul<Output = T> + Add<Output = T> + Copy + PartialOrd,
{
    if points.len() <= 3 {
        return points.to_vec();
    }

    let mut lower = Vec::with_capacity(points.len() / 2);
    for point in points {
        while lower.len() >= 2
            && !is_counter_clockwise(lower[lower.len() - 2], lower[lower.len() - 1], *point)
        {
            lower.pop();
        }
        lower.push(*point);
    }
    lower
}

/// Upper half of [`convex_hull`].
pub fn upper_convex_hull<T>(points: &[[T; 2]]) -> Vec<[T; 2]>
where
    T: Zero + Mul<Output = T> + Add<Output = T> + Copy + PartialOrd,
{
    if points.len() <= 3 {
        return points.to_vec();
    }

    let mut upper = Vec::with_capacity(points.len() / 2);
    for point in points.iter().rev() {
        while upper.len() >= 2
            && !is_counter_clockwise(upper[upper.len() - 2], upper[upper.len() - 1], *point)
        {
            upper.pop();
        }
        upper.push(*point);
    }
    upper
}

/// Check whether there is a counter-clockwise turn using the cross product of ca and cb interpreted as 3D vectors.
fn is_counter_clockwise<T>(a: [T; 2], b: [T; 2], c: [T; 2]) -> bool
where
    T: Zero + Mul<Output = T> + Add<Output = T> + Copy + PartialOrd,
{
    #[allow(clippy::suspicious_operation_groupings)]
    let positive = a[0] * b[1] + c[0] * c[1] + a[1] * c[0] + c[1] * b[0];
    #[allow(clippy::suspicious_operation_groupings)]
    let negative = a[0] * c[1] + c[0] * b[1] + a[1] * b[0] + c[1] * c[0];

    positive > negative
}

#[cfg(test)]
mod tests {
    use crate::voronoi::hull::lower_convex_hull;

    use super::convex_hull;

    #[test]
    fn test_convex_hull_of_triangle() {
        let points = [[0, 0], [0, 1], [1, 0]];
        assert_eq!(convex_hull(&points), points);
    }

    #[test]
    fn test_convex_hull_of_square() {
        let points = [[0, 0], [0, 1], [1, 0], [1, 1]];
        assert_eq!(
            convex_hull(&points),
            [points[0], points[2], points[3], points[1]]
        );
    }

    #[test]
    fn test_convex_hull_of_square_with_inscribed_point() {
        let points = [[0, 0], [0, 2], [1, 1], [2, 0], [2, 2]];
        assert_eq!(
            convex_hull(&points),
            [points[0], points[3], points[4], points[1]]
        );
    }

    #[test]
    fn test_lower_convex_hull_of_line_segments() {
        let points = [
            [0, 0],
            [0, 1],
            [1, 0],
            [2, 0],
            [3, 0],
            [3, 2],
            [4, 4],
            [5, 5],
            [6, 8],
        ];
        assert_eq!(
            lower_convex_hull(&points),
            [points[0], points[4], points[7], points[8]]
        );
    }
}
