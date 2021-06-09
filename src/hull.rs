use std::fmt::Debug;

use num_traits::PrimInt;

/// Andrew's monotone chain convex hull algorithm
///
/// https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
pub fn convex_hull<T: PrimInt + Debug>(points: &[[T; 2]]) -> Vec<[T; 2]> {
    let mut upper = Vec::with_capacity(points.len() / 2);
    let mut lower = Vec::with_capacity(points.len() / 2);
    for point in points {
        while lower.len() >= 2
            && !is_counter_clockwise(lower[lower.len() - 2], lower[lower.len() - 1], *point)
        {
            lower.pop();
        }
        lower.push(*point);
    }

    for point in points.iter().rev() {
        while upper.len() >= 2
            && !is_counter_clockwise(upper[upper.len() - 2], upper[upper.len() - 1], *point)
        {
            upper.pop();
        }
        upper.push(*point);
    }
    upper.pop();
    lower.pop();
    lower.append(&mut upper);
    lower
}

/// Check whether there is a counter-clockwise turn using the cross product of ca and cb interpreted as 3D vectors.
fn is_counter_clockwise<T: PrimInt + Debug>(a: [T; 2], b: [T; 2], c: [T; 2]) -> bool {
    let positive = a[0] * b[1] + c[0] * c[1] + a[1] * c[0] + c[1] * b[0];
    let negative = a[0] * c[1] + c[0] * b[1] + a[1] * b[0] + c[1] * c[0];
    positive
        .checked_sub(&negative)
        .map(|x| x >= T::zero())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::convex_hull;

    #[test]
    fn test_convex_hull_of_triangle() {
        let points = [[0, 0], [0, 1], [1, 0]];
        assert_eq!(convex_hull(&points), [points[0], points[2], points[1]]);
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
}
