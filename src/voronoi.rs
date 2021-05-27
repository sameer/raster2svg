use crate::abs_distance_squared;

pub fn compute_voronoi(vertices: &[[i64; 2]], width: usize, height: usize) -> Vec<Vec<[i64; 2]>> {
    if vertices.is_empty() {
        return vec![];
    }
    let expected_assignment_capacity = width * height / vertices.len();
    let mut point_assignments =
        vec![Vec::<[i64; 2]>::with_capacity(expected_assignment_capacity); vertices.len()];
    let width = width as i64;
    let height = height as i64;
    for point in (0..width)
        .map(|i| (0..height).map(move |j| [i, j]))
        .flatten()
    {
        let closest_vertex_to_point = vertices
            .iter()
            .enumerate()
            .min_by_key(|(_, vertex)| abs_distance_squared(point, **vertex))
            .unwrap();
        point_assignments[closest_vertex_to_point.0].push(*closest_vertex_to_point.1);
    }
    point_assignments
}
