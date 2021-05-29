use std::{cmp::Reverse, collections::BinaryHeap};

use crate::abs_distance_squared;
use fxhash::{FxHashMap as HashMap, FxHashSet as HashSet};

#[derive(PartialEq, Eq)]
struct BranchLeafEdgePair {
    branch_to_disconnected_node: [[i64; 2]; 2],
    leaf_to_leaf: [[i64; 2]; 2],
}

impl BranchLeafEdgePair {
    /// Best improvement to the tree (greatest reduction or smallest increase in length)
    fn diff(&self) -> i64 {
        abs_distance_squared(
            self.branch_to_disconnected_node[0],
            self.branch_to_disconnected_node[1],
        ) - abs_distance_squared(self.leaf_to_leaf[0], self.leaf_to_leaf[1])
    }
}

impl PartialOrd for BranchLeafEdgePair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.diff().partial_cmp(&other.diff())
    }
}

impl Ord for BranchLeafEdgePair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.diff().cmp(&other.diff())
    }
}

/// Approximate a TSP solution by way of all-pairs branch elimination
/// http://cs.uef.fi/sipu/pub/applsci-11-00177.pdf
pub fn approximate_tsp_with_mst(
    vertices: &[[i64; 2]],
    tree: &[[[i64; 2]; 2]],
) -> Vec<[[i64; 2]; 2]> {
    if vertices.len() <= 2 {
        return vec![];
    }
    let mut adjacency_map: HashMap<[i64; 2], Vec<[i64; 2]>> = HashMap::default();
    tree.iter().for_each(|edge| {
        adjacency_map.entry(edge[0]).or_default().push(edge[1]);
        adjacency_map.entry(edge[1]).or_default().push(edge[0]);
    });
    // let mut dfs_map: HashMap<[[i64; 2]; 2], HashSet<[i64; 2]>> = HashMap::default();
    // // Do a memoized DFS from each edge and its complementary directions
    // println!("memoized DFS");
    // for vertex in vertices {
    //     let mut stack = vec![vertex];
    //     let mut seen: HashSet<&[i64; 2]> = HashSet::default();
    //     let mut path: Vec<&[i64; 2]> = vec![vertex];
    //     while let Some(current) = stack.pop() {
    //         path.push(current);
    //         seen.insert(current);

    //         let neighbors = adjacency_map.get(current).unwrap();

    //         for neighbor in neighbors {
    //             if seen.contains(neighbor) {
    //                 continue;
    //             } else if dfs_map.contains_key(&[*current, *neighbor]) {
    //                 for (from, to) in path.iter().zip(path.iter().skip(1)) {
    //                     let from_to_updated = {
    //                         let current_to_neighbor_set =
    //                             dfs_map.get(&[*current, *neighbor]).unwrap();
    //                         if let Some(existing) = dfs_map.get(&[**from, **to]) {
    //                             existing
    //                                 .union(current_to_neighbor_set)
    //                                 .cloned()
    //                                 .collect::<HashSet<_>>()
    //                         } else {
    //                             current_to_neighbor_set.clone()
    //                         }
    //                     };
    //                     dfs_map.insert([**from, **to], from_to_updated);
    //                 }
    //             } else {
    //                 stack.push(neighbor);
    //             }
    //         }

    //         if neighbors.len() <= 1 {
    //             for (from, to) in path.iter().zip(path.iter().skip(1)) {
    //                 dfs_map.entry([**from, **to]).or_default().insert(*current);
    //             }
    //             path.pop();
    //         }
    //     }
    // }
    // println!("DFS done");
    let vertex_to_index = vertices
        .iter()
        .enumerate()
        .map(|(i, vertex)| (*vertex, i))
        .collect::<HashMap<_, _>>();

    loop {
        dbg!(adjacency_map
            .iter()
            .filter(|(_, vertex_edges)| vertex_edges.len() >= 3)
            .count());
        if let Some(BranchLeafEdgePair {
            branch_to_disconnected_node: [branch, disconnected_node],
            leaf_to_leaf: [branch_tree_leaf, disconnected_tree_leaf],
        }) = adjacency_map
            .iter()
            .filter(|(_, vertex_edges)| vertex_edges.len() >= 3)
            .map(|(branch, adjacencies)| {
                adjacencies.iter().map(move |adjacency| (branch, adjacency))
            })
            .flatten()
            .map(|(branch, disconnected_node_candidate)| {
                // Now there are (in theory) two disconnected trees
                // Find the two connected trees in the graph
                let mut branch_tree_visited: Vec<bool> = vec![false; vertices.len()];
                branch_tree_visited[*vertex_to_index.get(branch).unwrap()] = true;
                let mut branch_dfs = vec![branch];
                while let Some(head) = branch_dfs.pop() {
                    for adjacency in adjacency_map.get(head).unwrap() {
                        let adjacency_idx = *vertex_to_index.get(adjacency).unwrap();
                        // Explicitly skip this node to not enter the other connected component
                        if adjacency == disconnected_node_candidate {
                            continue;
                        } else if !branch_tree_visited[adjacency_idx] {
                            branch_tree_visited[adjacency_idx] = true;
                            branch_dfs.push(adjacency);
                        }
                    }
                }

                // Find leaves in the two
                // Pick the shortest possible link between two leaves that would reconnect the trees

                // let disconnected_tree_leaves = dfs_map
                //     .get(&[*branch, *disconnected_node_candidate])
                //     .unwrap()
                //     .iter()
                //     .filter(|disconnected_tree_vertex| {
                //         adjacency_map.get(*disconnected_tree_vertex).unwrap().len() == 1
                //     }).collect::<Vec<_>>();
                // let branch_tree_leaves = dfs_map
                //     .get(&[*disconnected_node_candidate, *branch])
                //     .unwrap()
                //     .iter()
                //     .filter(|branch_tree_vertex| {
                //         adjacency_map.get(*branch_tree_vertex).unwrap().len() == 1
                //     });

                // let (branch_tree_leaf, disconnected_tree_leaf) = branch_tree_leaves
                //     .map(|branch_tree_leaf| {
                //         disconnected_tree_leaves
                //             .iter()
                //             .map(move |disconnected_tree_leaf| {
                //                 (*branch_tree_leaf, *disconnected_tree_leaf)
                //             })
                //     })
                //     .flatten()
                //     .min_by_key(|(branch_tree_leaf, disconnected_tree_leaf)| {
                //         abs_distance_squared(*branch_tree_leaf, **disconnected_tree_leaf)
                //     })
                //     .unwrap();

                let disconnected_tree_leaves = branch_tree_visited
                    .iter()
                    .enumerate()
                    .filter_map(|(i, v)| if *v { None } else { Some(vertices[i]) })
                    .filter(|disconnected_tree_vertex| {
                        adjacency_map.get(disconnected_tree_vertex).unwrap().len() <= 1
                    })
                    .collect::<Vec<_>>();

                let (branch_tree_leaf, disconnected_tree_leaf) = branch_tree_visited
                    .iter()
                    .enumerate()
                    .filter_map(|(i, v)| if *v { Some(vertices[i]) } else { None })
                    .filter(|branch_tree_vertex| {
                        adjacency_map.get(branch_tree_vertex).unwrap().len() <= 1
                    })
                    .map(|branch_tree_leaf| {
                        disconnected_tree_leaves
                            .iter()
                            .map(move |disconnected_tree_leaf| {
                                (branch_tree_leaf, disconnected_tree_leaf)
                            })
                    })
                    .flatten()
                    .min_by_key(|(branch_tree_leaf, disconnected_tree_leaf)| {
                        abs_distance_squared(*branch_tree_leaf, **disconnected_tree_leaf)
                    })
                    .unwrap();
                BranchLeafEdgePair {
                    branch_to_disconnected_node: [*branch, *disconnected_node_candidate],
                    leaf_to_leaf: [branch_tree_leaf, *disconnected_tree_leaf],
                }
            })
            .max()
        {
            // Remove edge
            adjacency_map
                .get_mut(&branch)
                .unwrap()
                .retain(|adjacency| *adjacency != disconnected_node);
            adjacency_map
                .get_mut(&disconnected_node)
                .unwrap()
                .retain(|adjacency| *adjacency != branch);

            // Connect leaves
            adjacency_map
                .get_mut(&branch_tree_leaf)
                .unwrap()
                .push(disconnected_tree_leaf);
            adjacency_map
                .get_mut(&disconnected_tree_leaf)
                .unwrap()
                .push(branch_tree_leaf);
        } else {
            break;
        }
    }

    // MST is now a path, but needs to be made into a round trip
    let leaves: Vec<_> = adjacency_map
        .iter()
        .filter(|(_, vertex_edges)| vertex_edges.len() == 1)
        .map(|(vertex, _)| *vertex)
        .collect();
    if leaves.len() != 2 {
        unreachable!();
    }

    adjacency_map
        .get_mut(&leaves[0])
        .unwrap()
        .push(leaves[1].clone());
    adjacency_map
        .get_mut(&leaves[1])
        .unwrap()
        .push(leaves[0].clone());

    // Extract round trip from the adjacency list
    let mut path: Vec<[[i64; 2]; 2]> = Vec::with_capacity(vertices.len());
    path.push([
        *vertices.first().unwrap(),
        *adjacency_map
            .get(vertices.first().unwrap())
            .unwrap()
            .first()
            .unwrap(),
    ]);

    // The number of edges in a Hamiltonian cycle is equal to the number of vertices
    while path.len() < vertices.len() {
        let [second_to_last_vertex, last_vertex]: [[i64; 2]; 2] = *path.last().unwrap();
        let next_vertex = adjacency_map
            .get(&last_vertex)
            .unwrap()
            .iter()
            .find(|adjacency| **adjacency != second_to_last_vertex)
            .unwrap();
        path.push([last_vertex, *next_vertex]);
    }
    path
}

#[cfg(test)]
#[test]
fn tsp_is_correct_for_trivial_case() {
    let vertices = [[0, 0], [1, 1], [2, 2]];
    let tree = [[[0, 0], [1, 1]], [[1, 1], [2, 2]]];

    assert_eq!(
        approximate_tsp_with_mst(&vertices, &tree),
        &[[[0, 0], [1, 1]], [[1, 1], [2, 2]], [[2, 2], [0, 0]]]
    );
}

#[cfg(test)]
#[test]
fn tsp_is_correct_for_nontrivial_case() {
    let vertices: [[i64; 2]; 5] = [[24, 28], [371, 33], [235, 72], [157, 509], [156, 194]];
    let tree = [
        [[24, 28], [156, 194]],
        [[156, 194], [235, 72]],
        [[235, 72], [371, 33]],
        [[156, 194], [157, 509]],
    ];
    let distances = vertices
        .iter()
        .map(|vertex| {
            vertices
                .iter()
                .map(move |other_vertex| [vertex, other_vertex])
                .map(|[a, b]| {
                    (((a[0] - b[0]).pow(2) + (a[1] - b[1]).pow(2)) as f64)
                        .sqrt()
                        .round()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    println!("{:?}", distances);
    println!(
        "{:?}",
        approximate_tsp_with_mst(&vertices, &tree)
            .iter()
            .map(|[a, b]| {
                (((a[0] - b[0]).pow(2) + (a[1] - b[1]).pow(2)) as f64)
                    .sqrt()
                    .round()
            })
            .sum::<f64>()
    );
}
