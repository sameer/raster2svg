use num_traits::{FromPrimitive, PrimInt, Signed};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use spade::{
    delaunay::{DelaunayTreeLocate, IntDelaunayTriangulation},
    SpadeNum,
};
use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    fmt::Debug,
    hash::{BuildHasherDefault, Hash},
};

use crate::math::abs_distance_squared;

#[derive(PartialEq, Eq, Hash, Debug)]
struct PriorityQueueEdge<T: PrimInt + PartialEq + Eq + PartialOrd + Ord> {
    from: [T; 2],
    to: [T; 2],
}

impl<T: PrimInt + Signed + PartialEq + Eq + PartialOrd + Ord + Debug> PartialOrd
    for PriorityQueueEdge<T>
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(
            abs_distance_squared(self.from, self.to)
                .cmp(&abs_distance_squared(other.from, other.to)),
        )
    }
}

impl<T: PrimInt + Signed + PartialEq + Eq + PartialOrd + Ord + Debug> Ord for PriorityQueueEdge<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Given the delaunay triangulation of 2D points, compute the MST with Prim's algorithm in O(E log(v)) time.
///
/// <https://en.wikipedia.org/wiki/Euclidean_minimum_spanning_tree#Algorithms_for_computing_EMSTs_in_two_dimensions>
pub fn compute_mst<T>(
    points: &[[T; 2]],
    delaunay: &IntDelaunayTriangulation<[T; 2], DelaunayTreeLocate<[T; 2]>>,
) -> Vec<[[T; 2]; 2]>
where
    T: PrimInt + Signed + FromPrimitive + Hash + Debug + SpadeNum,
{
    let edges_by_vertex: HashMap<_, Vec<_>> = delaunay
        .vertices()
        .map(|vertex| {
            let from = *vertex;
            (
                from,
                vertex
                    .ccw_out_edges()
                    .map(|edge| {
                        let to = *edge.to();
                        PriorityQueueEdge { from, to }
                    })
                    .collect(),
            )
        })
        .collect();

    let mut in_mst = HashSet::with_capacity_and_hasher(points.len(), BuildHasherDefault::default());
    let mut edge_priority_queue = BinaryHeap::new();

    // Kickstart MST with 1 vertex
    if !points.is_empty() {
        let first_vertex = points[0];
        in_mst.insert(first_vertex);
        edge_priority_queue.extend(edges_by_vertex[&first_vertex].iter().map(Reverse));
    }

    let mut mst = Vec::with_capacity(points.len().saturating_sub(1));
    while let Some(shortest_edge) = edge_priority_queue.pop() {
        // Claim: we know the "from" of the shortest edge will always be
        // in the MST, because all edges in the priority queue point
        // outwards from the tree built so far.
        match in_mst.contains(&shortest_edge.0.to) {
            // Edge would not add a new point to the MST
            true => continue,
            // Add edges introduced by new vertex
            false => {
                let to = &shortest_edge.0.to;
                in_mst.insert(*to);
                mst.push([shortest_edge.0.from, shortest_edge.0.to]);

                edge_priority_queue.extend(
                    edges_by_vertex[&shortest_edge.0.to]
                        .iter()
                        .filter(|edge| !in_mst.contains(&edge.to))
                        .map(Reverse),
                );
                if mst.len() == points.len().saturating_sub(1) {
                    // Early stopping condition, MST already has all the edges
                    break;
                }
            }
        }
    }
    mst
}

#[cfg(test)]
#[test]
fn test_mst_is_correct_for_trivial_case() {
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
