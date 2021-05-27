use fxhash::{FxHashMap as HashMap, FxHashSet as HashSet};
use spade::delaunay::{DelaunayTreeLocate, IntDelaunayTriangulation};
use std::{cmp::Reverse, collections::BinaryHeap};

#[derive(PartialEq, Eq, Hash)]
struct PriorityQueueEdge {
    from: [i64; 2],
    to: [i64; 2],
}

impl PartialOrd for PriorityQueueEdge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(
            crate::abs_distance_squared(self.from, self.to)
                .cmp(&crate::abs_distance_squared(other.from, other.to)),
        )
    }
}

impl Ord for PriorityQueueEdge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        crate::abs_distance_squared(self.from, self.to)
            .cmp(&crate::abs_distance_squared(other.from, other.to))
    }
}

/// https://en.wikipedia.org/wiki/Euclidean_minimum_spanning_tree#Algorithms_for_computing_EMSTs_in_two_dimensions
/// Given a delaunay triangulation, compute the MST
pub fn compute_mst(
    vertices: &[[i64; 2]],
    delaunay: &IntDelaunayTriangulation<[i64; 2], DelaunayTreeLocate<[i64; 2]>>,
) -> Vec<[[i64; 2]; 2]> {
    let mut edges_by_vertex: HashMap<[i64; 2], Vec<_>> = HashMap::default();
    delaunay.edges().for_each(|edge| {
        let from: &[i64; 2] = &edge.from();
        let to: &[i64; 2] = &edge.to();
        edges_by_vertex.entry(from.clone()).or_default().push(edge);
        edges_by_vertex.entry(to.clone()).or_default().push(edge);
    });

    let mut in_mst: HashSet<[i64; 2]> = HashSet::default();
    let mut edge_priority_queue = BinaryHeap::new();

    // Kickstart MST with 1 vertex
    if !vertices.is_empty() {
        let first_vertex: &[i64; 2] = &vertices[0];
        in_mst.insert(first_vertex.clone());
        for edge in edges_by_vertex.get(first_vertex).unwrap() {
            let from: &[i64; 2] = &edge.from();
            let to: &[i64; 2] = &edge.to();
            edge_priority_queue.push(Reverse(PriorityQueueEdge {
                from: from.clone(),
                to: to.clone(),
            }));
        }
    }

    let mut mst = vec![];
    while mst.len() < vertices.len().saturating_sub(1) {
        while let Some(shortest_edge) = edge_priority_queue.pop() {
            match (
                in_mst.contains(&shortest_edge.0.from),
                in_mst.contains(&shortest_edge.0.to),
            ) {
                (true, true) => continue,
                (false, true) => {
                    let from: &[i64; 2] = &shortest_edge.0.from;
                    in_mst.insert(from.clone());
                    mst.push([shortest_edge.0.to, shortest_edge.0.from]);

                    for edge in edges_by_vertex.get(&shortest_edge.0.from).unwrap() {
                        let from: &[i64; 2] = &edge.from();
                        let to: &[i64; 2] = &edge.to();
                        edge_priority_queue.push(Reverse(PriorityQueueEdge {
                            from: from.clone(),
                            to: to.clone(),
                        }));
                    }
                }
                (true, false) => {
                    let to: &[i64; 2] = &shortest_edge.0.to;
                    in_mst.insert(to.clone());
                    mst.push([shortest_edge.0.from, shortest_edge.0.to]);

                    for edge in edges_by_vertex.get(&shortest_edge.0.to).unwrap() {
                        let from: &[i64; 2] = &edge.from();
                        let to: &[i64; 2] = &edge.to();
                        edge_priority_queue.push(Reverse(PriorityQueueEdge {
                            from: from.clone(),
                            to: to.clone(),
                        }));
                    }
                }
                (false, false) => unreachable!(),
            }
        }
    }
    mst
}

#[cfg(test)]
#[test]
fn mst_is_correct_for_trivial_case() {
    let mut delaunay = IntDelaunayTriangulation::new();
    let points = [[0, 0], [1, 1], [2, 2]];
    for point in &points {
        delaunay.insert(*point);
    }
    assert_eq!(
        compute_mst(&points, &delaunay),
        &[[[0, 0], [1, 1]], [[1, 1], [2, 2]]]
    );
}
